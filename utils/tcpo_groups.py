"""Exclusive question-group proportions aligned with the complete rollout batch."""
import torch


def trajectory_group_fractions(state, prompt_indices, *, expected_count):
    state = torch.as_tensor(state).detach().cpu()
    indices = torch.as_tensor(prompt_indices).detach().cpu()
    if (state.dtype != torch.int64 or indices.dtype != torch.int64 or state.ndim != 1
            or indices.shape != state.shape or not state.numel()
            or (state < 0).any() or (state > 2).any() or (indices < 0).any()):
        raise ValueError('TCPO requires aligned integer trajectory labels and question indices.')
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('TCPO group rollout count must be a positive integer.')
    _, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse * 3 + state, minlength=(int(inverse.max())+1)*3).reshape(-1, 3)
    sizes = counts.sum(1)
    if not (sizes == expected_count).all():
        raise ValueError('TCPO requires every complete question group before Learner sharding.')
    return (counts.to(torch.float64) / sizes[:, None])[inverse]
