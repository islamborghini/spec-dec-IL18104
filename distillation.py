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


def restore_training_state(optimizer, path, expected_step):
    """Resume Adam moments and RNG only from a checkpoint at the declared update."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["optimizer_step"] != expected_step:
        raise ValueError("Optimizer checkpoint step does not match adapter/history")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"])
    return state
