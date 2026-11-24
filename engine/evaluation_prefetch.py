"""Generate one next training batch during the final evaluation dataset."""
from __future__ import annotations

import pickle
import time
import uuid

from utils.eval import write_json


class EvaluationTrainPrefetch:
    def __init__(self, controller, client, pending, issued, step):
        if pending:
            raise RuntimeError("Evaluation prefetch requires a drained training boundary.")
        self.controller, self.client, self.pending = controller, client, pending
        self.issued, self.step = issued, step
        self.envelopes, self.dispatched = {}, set()
        self.round_id = None
        self.audit = {"evaluation_step": step, "policy_version": controller._last_policy_version,
                      "dispatches": [], "status": "evaluating"}
        self.path = controller.output_dir / "evaluation_prefetch" / f"step_{step:06d}_{uuid.uuid4().hex[:8]}.json"

    def actor_idle(self, gpu_id):
        """Called only when this card has no remaining evaluation requests."""
        if gpu_id in self.dispatched:
            raise RuntimeError("An evaluation Actor was declared idle twice.")
        if self.round_id is None:
            self.issued = self.controller._push_training_group(
                self.client, self.pending, count=1, issued=self.issued, hold_learner=True,
                actor_dispatch=lambda gpu, envelope: self.envelopes.__setitem__(gpu, envelope))
            self.round_id = self.controller._active_round_id
            self.audit.update(round_id=self.round_id, batch_ids=list(self.pending),
                              training_actor_gpu_ids=sorted(self.envelopes))
        self.dispatched.add(gpu_id)
        envelope = self.envelopes.pop(gpu_id, None)
        if envelope is not None:
            self.client.lpush(f"{self.controller.queue_keys()['actor_train']}:gpu:{gpu_id}",
                              pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
            now = time.monotonic()
            self.audit["dispatches"].append({"gpu_id": gpu_id, "monotonic": now})
            print(f"[Controller] Evaluation Actor GPU {gpu_id} started next-batch rollouts; "
                  f"round={self.round_id}, Learners held until all evaluation finishes.", flush=True)
        write_json(self.path, self.audit)

    def finish(self, *, release):
        self.audit["evaluation_returned_monotonic"] = time.monotonic()
        if self.round_id is None:
            return
        if release:
            if self.envelopes:
                raise RuntimeError("Evaluation finished before every training shard was dispatched.")
            gate = self.controller.gpu_gates.round_gate
            self.audit["actor_finished_at_evaluation_end"] = gate.actor_finish_times(self.round_id)
            self.audit["all_training_rollouts_ready_at_evaluation_end"] = all(
                0 < self.audit["actor_finished_at_evaluation_end"][gpu] <= self.audit["evaluation_returned_monotonic"]
                for gpu in self.audit["training_actor_gpu_ids"])
            self.audit["learner_release_monotonic"] = time.monotonic()
            self.audit["status"] = "released"
            # Commit the trace before opening the process-shared gate.
            write_json(self.path, self.audit)
            gate.release_learner(self.round_id)
            finished = [value for value in self.audit["actor_finished_at_evaluation_end"].values()
                        if 0 < value <= self.audit["evaluation_returned_monotonic"]]
            self.controller._swanlab.log_components({"evaluation": {"prefetch": {
                "dispatched_actor_shards": len(self.audit["dispatches"]),
                "completed_actor_shards": len(finished),
                "all_rollouts_ready": int(self.audit["all_training_rollouts_ready_at_evaluation_end"]),
                "max_rollout_lead_seconds": max((self.audit["evaluation_returned_monotonic"]-value
                                                 for value in finished), default=0.),
                "learner_released": 1}}}, step=self.step)
        else:
            # Shutdown sets worker stop events. Leave the barrier closed so
            # prefetched data can never update a policy rejected by evaluation.
            self.audit["status"] = "discarded"
            write_json(self.path, self.audit)
            self.controller._swanlab.log_components({"evaluation": {"prefetch": {
                "learner_released": 0, "discarded": 1}}}, step=self.step)
