"""Probability distillation on unpadded next-token distributions, temperature 1."""


def distillation_loss(student_logits, teacher_logits):
    import torch.nn.functional as F

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Expected matching (batch, tokens, vocabulary) logits")
    if student_logits.shape[0] * student_logits.shape[1] == 0:
        raise ValueError("Empty token batch")
    teacher_logp = teacher_logits.detach().float().log_softmax(-1)
    student_logp = student_logits.float().log_softmax(-1)
    return F.kl_div(student_logp, teacher_logp, log_target=True, reduction="sum") / (
        student_logits.shape[0] * student_logits.shape[1]
    )
