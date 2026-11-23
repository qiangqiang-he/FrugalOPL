"""User-specified per-run TCDO weight-halving decisions."""


def tuning_decision(evaluations, completed_steps, incorrect_weight, truncated_weight, stop_reason=None,
                    *, target_steps=100, min_eval_step=10):
    post = [e for e in evaluations if e["step"] >= min_eval_step]
    rates = [e["datasets"]["aime24"]["metrics"]["truncation_rate"] for e in post]
    bound_exceeded = bool(stop_reason and stop_reason.get('bound')=='lower'
        and stop_reason.get('dataset')=='aime24' and stop_reason.get('metric')=='truncation_rate'
        and stop_reason.get('step',0)>=min_eval_step and stop_reason.get('expected_response_count',0)>0
        and stop_reason['known_truncated_response_count']/stop_reason['expected_response_count']>0.5)
    if bound_exceeded or any(rate > 0.5 for rate in rates):
        return {"action": "halve_incorrect", "incorrect_weight": incorrect_weight / 2,
                "truncated_weight": truncated_weight}
    if completed_steps == target_steps and rates and all(rate < 0.1 for rate in rates):
        return {"action": "halve_truncated", "incorrect_weight": incorrect_weight,
                "truncated_weight": truncated_weight / 2}
    return {"action": "accept" if completed_steps == target_steps else "incomplete",
            "incorrect_weight": incorrect_weight, "truncated_weight": truncated_weight}
