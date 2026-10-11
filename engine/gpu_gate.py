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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class _StopEvent(Protocol):
    """The small Event surface required while a Worker waits for a gate."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


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

    def __init__(self, gates: Mapping[int, GpuGate]) -> None:
        self._gates = dict(gates)

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


__all__ = ["GpuGate", "GpuGateRegistry"]
