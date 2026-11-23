"""Opt-in graceful stop at a completed evaluation boundary."""
import math


def evaluation_stop_reason(rule, evaluation):
    if not rule:
        return None
    minimum = rule.get("min_step", 1)
    above = rule["above"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("evaluation_stop.min_step must be positive; the baseline never triggers stopping")
    if isinstance(above, bool) or not isinstance(above, (float, int)) or not math.isfinite(above):
        raise ValueError("evaluation_stop.above must be finite")
    if evaluation["step"] < minimum:
        return None
    value = evaluation["datasets"][rule["dataset"]]["metrics"][rule["metric"]]
    if value > above:
        return {"step": evaluation["step"], "dataset": rule["dataset"], "metric": rule["metric"],
                "value": value, "above": above}
    return None


def partial_evaluation_stop_reason(rule,step,dataset,progress):
    if not rule or not rule.get('allow_partial_lower_bound',False):
        return None
    if rule['dataset']!=dataset or rule['metric']!='truncation_rate' or progress['complete']:
        return None
    expected=progress['expected_response_count']
    received=progress['received_response_count']
    cut=progress['known_truncated_response_count']
    if not 0<=cut<=received<=expected or expected<=0:
        raise ValueError('Invalid partial evaluation counts')
    bound=cut/expected
    reason=evaluation_stop_reason(rule,{'step':step,'datasets':{dataset:{'metrics':{'truncation_rate':bound}}}})
    if reason:
        return {**reason,'complete':False,'bound':'lower','expected_response_count':expected,
                'received_response_count':received,'known_truncated_response_count':cut,
                'known_correct_response_count':progress['known_correct_response_count']}
    return None
