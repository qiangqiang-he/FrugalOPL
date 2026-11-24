"""Cooperative execution gates for Workers that share a physical GPU.

The framework keeps model residency independent from execution ownership: an
Actor, Teacher, and Learner may all have models resident on one local GPU,
but only one Worker is allowed to issue its GPU work at a time.  A Controller
creates this registry once and passes its process-shared semaphores to the
Worker Managers.  Workers, not the Controller, acquire and release a gate
around a bounded processing turn.

Only the physical GPU id is meaningful here.  Worker children set
``CUDA_VISIBLE_DEVICES`` and therefore see their assigned physical card as
logical CUDA device zero; using that logical index for synchronization would
incorrectly merge unrelated cards.
"""

from __future__ import annotations

import multiprocessing as mp
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class _StopEvent(Protocol):
    """The small Event surface required while a Worker waits for a gate."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True)
class RoundManifest:
    round_id: int
    batch_ids: tuple[str, ...]
    policy_version: int


class SharedRoundGate:
    """CPU-only phase handoff for a complete co-located Worker group.

    This is not a binary execution lock: all Actors may run together during
    ACTOR, and all Learner ranks may run together during LEARNER. Per-device
    gates still protect physical execution against other Worker types.
    """

    _NAMES = {0: "idle", 1: "actor", 2: "learner", 3: "complete", -1: "aborted"}

    def __init__(self, actor_gpu_ids: tuple[int, ...], learner_ranks: int, max_batches: int, *, ctx=None):
        if not actor_gpu_ids or learner_ranks < 1 or max_batches < 1:
            raise ValueError("A shared round requires Actors, Learner ranks and a positive batch limit.")
        context = ctx if ctx is not None else mp.get_context()
        self.actor_gpu_ids = tuple(actor_gpu_ids)
        self.learner_ranks = learner_ranks
        self.max_batches = max_batches
        self._condition = context.Condition()
        self._phase = context.Value("i", 0, lock=False)
        self._round = context.Value("q", 0, lock=False)
        self._manifest = context.Array("c", max(16384, max_batches * 256), lock=False)
        self._error = context.Array("c", 4096, lock=False)
        self._actor_done = context.Array("b", len(actor_gpu_ids), lock=False)
        self._actor_finished_at = context.Array("d", len(actor_gpu_ids), lock=False)
        self._learner_ready = context.Array("b", learner_ranks, lock=False)
        self._learner_done = context.Array("b", learner_ranks, lock=False)
        self._learner_blocked = context.Value("b", False, lock=False)

    @property
    def learner_blocked(self) -> bool:
        with self._condition:
            return bool(self._learner_blocked.value)

    @property
    def phase(self) -> str:
        with self._condition:
            return self._NAMES[self._phase.value]

    def _check(self) -> None:
        if self._phase.value == -1:
            raise RuntimeError("Shared GPU round aborted: " + self._error.value.decode("utf-8", errors="replace"))

    @staticmethod
    def _stopped(stop_event) -> bool:
        check = getattr(stop_event, "is_set", None)
        return bool(callable(check) and check())

    def _read_manifest(self) -> RoundManifest:
        value = json.loads(self._manifest.value.decode("utf-8"))
        return RoundManifest(value["round_id"], tuple(value["batch_ids"]), value["policy_version"])

    def begin_round(self, batch_ids, policy_version: int, actor_gpu_ids=None, *, hold_learner=False) -> RoundManifest:
        if not isinstance(hold_learner, bool):
            raise ValueError("hold_learner must be boolean.")
        ids = tuple(batch_ids)
        if not ids or len(ids) > self.max_batches or len(set(ids)) != len(ids):
            raise ValueError("Round batch IDs must be non-empty, unique and within the in-flight limit.")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("Round batch IDs must be non-empty strings.")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("Round policy_version must be non-negative.")
        active = set(self.actor_gpu_ids if actor_gpu_ids is None else actor_gpu_ids)
        if not active or not active <= set(self.actor_gpu_ids):
            raise ValueError("Round Actor IDs must belong to the configured Actor group.")
        with self._condition:
            self._check()
            if self._phase.value not in (0, 3):
                raise RuntimeError("Cannot open a round before the preceding round has fully released its GPUs.")
            manifest = RoundManifest(self._round.value + 1, ids, policy_version)
            data = json.dumps(manifest.__dict__).encode("utf-8")
            if len(data) >= len(self._manifest):
                raise ValueError("Round manifest exceeds the shared control buffer.")
            self._manifest.value = data
            self._round.value = manifest.round_id
            self._learner_blocked.value = hold_learner
            for index, gpu_id in enumerate(self.actor_gpu_ids):
                self._actor_done[index] = int(gpu_id not in active)
                self._actor_finished_at[index] = 0.0
            for index in range(self.learner_ranks):
                self._learner_ready[index] = self._learner_done[index] = 0
            self._phase.value = 1
            self._condition.notify_all()
            return manifest

    def release_learner(self, round_id: int) -> None:
        """Release an evaluation boundary only after every dataset is complete."""
        with self._condition:
            self._check()
            if self._round.value != round_id or self._phase.value not in (1, 2) or not self._learner_blocked.value:
                raise RuntimeError("Learner release does not match a held generation round.")
            self._learner_blocked.value = False
            self._condition.notify_all()

    def wait_actor(self, gpu_id: int, round_id: int, stop_event=None) -> bool:
        if gpu_id not in self.actor_gpu_ids:
            raise ValueError("Unknown Actor GPU.")
        with self._condition:
            while True:
                self._check()
                if self._stopped(stop_event):
                    return False
                if self._round.value == round_id and self._phase.value == 1:
                    return True
                if self._round.value >= round_id:
                    raise RuntimeError("Actor requested an expired round grant.")
                self._condition.wait(0.1)

    def actor_done(self, gpu_id: int, round_id: int) -> None:
        with self._condition:
            self._check()
            if self._round.value != round_id or self._phase.value != 1:
                raise RuntimeError("Actor acknowledgement does not match the active generation round.")
            index = self.actor_gpu_ids.index(gpu_id)
            if self._actor_done[index]:
                raise RuntimeError("Duplicate Actor round acknowledgement.")
            self._actor_done[index] = 1
            self._actor_finished_at[index] = time.monotonic()
            if all(self._actor_done):
                self._phase.value = 2
            self._condition.notify_all()

    def wait_local_learner(self, gpu_id, after_round=0, stop_event=None, *, timeout=1.0):
        """Admit a Learner as soon as its own Actor has released the card."""
        index = self.actor_gpu_ids.index(gpu_id)
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._check()
                if self._stopped(stop_event):
                    return None
                if (not self._learner_blocked.value and self._phase.value in (1, 2)
                        and self._round.value > after_round and self._actor_done[index]):
                    return self._read_manifest()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(0.1, remaining))

    def actor_finish_times(self, round_id):
        with self._condition:
            self._check()
            if self._round.value != round_id:
                raise RuntimeError("Actor timing requested for a different round.")
            return dict(zip(self.actor_gpu_ids, self._actor_finished_at[:]))

    def wait_learner(self, after_round: int = 0, stop_event=None, *, timeout: float | None = None) -> RoundManifest | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while True:
                self._check()
                if self._stopped(stop_event):
                    return None
                if not self._learner_blocked.value and self._phase.value == 2 and self._round.value > after_round:
                    return self._read_manifest()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(0.1 if remaining is None else min(0.1, remaining))

    def learner_ready(self, rank: int, round_id: int, stop_event=None, *, wait_for_actors=False) -> bool:
        if rank not in range(self.learner_ranks):
            raise ValueError("Unknown Learner rank.")
        with self._condition:
            self._check()
            while wait_for_actors and self._round.value == round_id and self._phase.value == 1:
                if self._stopped(stop_event):
                    return False
                self._condition.wait(0.1)
                self._check()
            if self._round.value != round_id or self._phase.value != 2 or self._learner_blocked.value:
                raise RuntimeError("Learner readiness does not match the active round.")
            if self._learner_ready[rank]:
                raise RuntimeError("Duplicate Learner readiness acknowledgement.")
            self._learner_ready[rank] = 1
            self._condition.notify_all()
            while not all(self._learner_ready):
                self._check()
                if self._stopped(stop_event):
                    return False
                self._condition.wait(0.1)
            self._check()
            if self._stopped(stop_event):
                return False
            return True

    def learner_done(self, rank: int, round_id: int) -> None:
        if rank not in range(self.learner_ranks):
            raise ValueError("Unknown Learner rank.")
        with self._condition:
            self._check()
            if self._round.value != round_id or self._phase.value != 2 or not all(self._learner_ready):
                raise RuntimeError("Learner completion does not match a fully admitted training round.")
            if self._learner_done[rank]:
                raise RuntimeError("Duplicate Learner completion acknowledgement.")
            self._learner_done[rank] = 1
            if all(self._learner_done):
                self._phase.value = 3
            self._condition.notify_all()

    def wait_complete(self, round_id: int, *, timeout: float = 0.1) -> bool:
        with self._condition:
            self._check()
            if self._round.value == round_id and self._phase.value == 3:
                return True
            self._condition.wait(timeout)
            self._check()
            return self._round.value == round_id and self._phase.value == 3

    def abort(self, reason: str = "run stopped") -> None:
        with self._condition:
            self._error.value = str(reason).encode("utf-8")[:4000]
            self._phase.value = -1
            self._condition.notify_all()


@dataclass(frozen=True)
class GpuGate:
    """One physical-GPU execution permit.

    ``semaphore is None`` represents an exclusive GPU.  Such a gate is
    deliberately a no-op: the common 4/8-GPU dedicated-worker path should
    not inherit a lock, a polling delay, or any scheduling overhead.
    """

    physical_gpu_id: int
    semaphore: object | None = None

    @property
    def is_shared(self) -> bool:
        """Whether more than one Worker type was assigned this physical GPU."""

        return self.semaphore is not None

    def acquire(
        self,
        stop_event: _StopEvent | object | None = None,
        *,
        poll_interval_sec: float = 0.1,
    ) -> bool:
        """Acquire the GPU permit, returning ``False`` when shutdown wins.

        A shared multiprocessing semaphore is polled with a short timeout so
        a Worker blocked behind another Worker can still observe its stop
        event.  Minimal CPU doubles sometimes expose only ``acquire()``;
        those retain the same semantics without a timeout argument.
        """

        if not self.is_shared:
            return True
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive.")
        semaphore = self.semaphore
        if semaphore is None:  # pragma: no cover - narrowed by is_shared.
            return True
        is_set = getattr(stop_event, "is_set", None)
        acquire = getattr(semaphore, "acquire", None)
        if not callable(acquire):
            raise TypeError("GpuGate semaphore must provide acquire().")

        while True:
            if callable(is_set) and is_set():
                return False
            try:
                acquired = acquire(timeout=poll_interval_sec)
            except TypeError:
                # Test doubles and a few minimal synchronization adapters do
                # not accept the timeout keyword.  They are expected to be
                # non-blocking for their test use.
                acquired = acquire()
            if acquired is not False:
                return True

    def release(self) -> None:
        """Release a previously acquired shared permit; exclusive gates noop."""

        if not self.is_shared:
            return
        semaphore = self.semaphore
        release = getattr(semaphore, "release", None)
        if not callable(release):
            raise TypeError("GpuGate semaphore must provide release().")
        release()

    def release_then_yield(
        self,
        stop_event: _StopEvent | object | None,
        *,
        delay_sec: float = 2.0,
    ) -> None:
        """Release first, then give a competing shared-GPU Worker a chance.

        The delay is intentionally outside the critical section.  It provides
        bounded cooperative fairness after one Worker has completed its
        ``max_batches_per_turn`` without penalizing a GPU dedicated to a
        single Worker.
        """

        if delay_sec < 0:
            raise ValueError("delay_sec must be non-negative.")
        self.release()
        self.yield_after_release(stop_event, delay_sec=delay_sec)

    def yield_after_release(
        self,
        stop_event: _StopEvent | object | None,
        *,
        delay_sec: float = 2.0,
    ) -> None:
        """Cooperatively wait after a caller has already released the permit."""

        if delay_sec < 0:
            raise ValueError("delay_sec must be non-negative.")
        if not self.is_shared or delay_sec == 0:
            return
        wait = getattr(stop_event, "wait", None)
        if callable(wait):
            wait(delay_sec)
            return
        time.sleep(delay_sec)


class GpuGateRegistry:
    """Map physical GPU IDs to shared or no-op :class:`GpuGate` objects."""

    def __init__(self, gates: Mapping[int, GpuGate], *, round_gate: SharedRoundGate | None = None) -> None:
        self._gates = dict(gates)
        self.round_gate = round_gate

    @classmethod
    def from_worker_gpu_ids(
        cls,
        worker_gpu_ids: Mapping[str, Sequence[object]],
        *,
        ctx: object | None = None,
    ) -> "GpuGateRegistry":
        """Create one permit for each physical GPU used by multiple Workers.

        ``worker_gpu_ids`` is keyed by Worker type (``actor``, ``teacher``,
        ``learner``).  Repeated entries within one Worker type count once;
        a card becomes shared only when two distinct Worker types name it.
        """

        if not isinstance(worker_gpu_ids, Mapping):
            raise TypeError("worker_gpu_ids must be a mapping of worker names to GPU-id sequences.")
        users: dict[int, set[str]] = {}
        for worker_name, raw_gpu_ids in worker_gpu_ids.items():
            if not isinstance(worker_name, str) or not worker_name:
                raise ValueError("Worker names must be non-empty strings.")
            if isinstance(raw_gpu_ids, (str, bytes)) or not isinstance(raw_gpu_ids, Sequence):
                raise TypeError(f"GPU ids for Worker {worker_name!r} must be a sequence.")
            normalized: set[int] = set()
            for value in raw_gpu_ids:
                if isinstance(value, bool):
                    raise ValueError("GPU ids must be non-negative integers.")
                try:
                    gpu_id = int(value)
                except (TypeError, ValueError) as error:
                    raise ValueError("GPU ids must be non-negative integers.") from error
                if gpu_id < 0:
                    raise ValueError("GPU ids must be non-negative integers.")
                normalized.add(gpu_id)
            for gpu_id in normalized:
                users.setdefault(gpu_id, set()).add(worker_name)

        context = ctx if ctx is not None else mp.get_context()
        semaphore_factory = getattr(context, "Semaphore", None)
        gates: dict[int, GpuGate] = {}
        for gpu_id, worker_names in users.items():
            if len(worker_names) <= 1:
                gates[gpu_id] = GpuGate(physical_gpu_id=gpu_id)
                continue
            if not callable(semaphore_factory):
                raise TypeError("GpuGateRegistry context must provide Semaphore().")
            gates[gpu_id] = GpuGate(physical_gpu_id=gpu_id, semaphore=semaphore_factory(1))
        return cls(gates)

    def for_gpu(self, physical_gpu_id: int) -> GpuGate:
        """Return the stable Gate object for a configured physical GPU."""

        if isinstance(physical_gpu_id, bool) or not isinstance(physical_gpu_id, int):
            raise ValueError("physical_gpu_id must be a non-negative integer.")
        if physical_gpu_id < 0:
            raise ValueError("physical_gpu_id must be a non-negative integer.")
        try:
            return self._gates[physical_gpu_id]
        except KeyError as error:
            raise KeyError(f"No GPU gate was registered for physical GPU {physical_gpu_id}.") from error

    @property
    def shared_gpu_ids(self) -> tuple[int, ...]:
        """Sorted physical GPU IDs that have a real process-shared semaphore."""

        return tuple(sorted(gpu_id for gpu_id, gate in self._gates.items() if gate.is_shared))


__all__ = ["GpuGate", "GpuGateRegistry", "RoundManifest", "SharedRoundGate"]
