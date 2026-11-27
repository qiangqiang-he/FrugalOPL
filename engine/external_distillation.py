"""Response-aligned Forward KL from an independent frozen co-located Teacher."""
import math

import torch
from torch.utils.checkpoint import checkpoint

from utils.external_teacher import validate_teacher_messages
from utils.forward_kl import forward_kl_token_values


def external_forward_kl(learner, batch, *, teacher_messages, temperature=1.):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('External Forward KL temperature must be positive and finite.')
    if not torch.is_grad_enabled():
        raise RuntimeError('External Forward KL requires Student gradients.')
    if not isinstance(teacher_messages, (list, tuple)) or len(teacher_messages) != 1:
        raise ValueError('External Forward KL scores exactly one rollout at a time.')
    validate_teacher_messages(teacher_messages[0])
    teacher = learner.external_teacher
    if teacher is None or teacher.model is None:
        raise RuntimeError('External Forward KL requires an initialized independent Teacher.')
    if any(p.requires_grad for p in teacher.model.parameters()) or teacher.model.training:
        raise RuntimeError('External Teacher must remain frozen and in eval mode.')
    if next(teacher.model.parameters()).device != learner.device:
        raise RuntimeError('External Teacher and Student must use the same Learner device.')
    student = learner._module()
    student_batch = learner._prepare_forward_batch(batch, prompts=None, prompt_ids=None)
    _prompt, teacher_ids = teacher.prompt_ids(teacher_messages[0])
    teacher_batch = learner._prepare_forward_batch(batch, prompts=None, prompt_ids=[teacher_ids])
    teacher_limit = getattr(teacher.model.config, 'max_position_embeddings', None)
    if teacher_limit is not None and teacher_batch['input_ids'].shape[1] > teacher_limit:
        raise ValueError('External Teacher sequence exceeds its model context limit.')
    length = student_batch['response_ids'].shape[1]
    student_start = student_batch['prompt_ids'].shape[1]-1
    teacher_start = teacher_batch['prompt_ids'].shape[1]-1
    prior_training = bool(student.training)
    teacher_head, student_head = learner._lm_head(teacher.model), learner._lm_head(student)
    with torch.no_grad():
        teacher_hidden = learner._forward_hidden(teacher.model, input_ids=teacher_batch['input_ids'],
            attention_mask=teacher_batch['attention_mask'], position_ids=teacher_batch['position_ids'])
        teacher_hidden = teacher_hidden[:, teacher_start:teacher_start+length].detach()
    keys = ('response_forward_kl', 'response_forward_kl_unclipped',
            'response_forward_kl_clipped_fraction', 'response_student_entropy_nats', 'response_teacher_entropy_nats')

    def score(student_hidden, frozen_hidden):
        student_logits = student_head(student_hidden)
        with torch.no_grad():
            teacher_logits = teacher_head(frozen_hidden)
        values = forward_kl_token_values(student_logits, teacher_logits,
                                        temperature=temperature, token_clip=None, top_k=learner.kl_top_k)
        return tuple(values[key] for key in keys)

    try:
        student.train()
        hidden = learner._forward_hidden(student, input_ids=student_batch['input_ids'],
            attention_mask=student_batch['attention_mask'], position_ids=student_batch['position_ids'])
        student_hidden = hidden[:, student_start:student_start+length]
        values = checkpoint(score, student_hidden, teacher_hidden, use_reentrant=False)
        return {key: value if i == 0 else value.detach() for i, (key, value) in enumerate(zip(keys, values, strict=True))}
    finally:
        student.train(prior_training)
