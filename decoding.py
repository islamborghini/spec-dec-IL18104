"""Batch-one greedy target and speculative decoding with reusable KV caches.

This implements temperature-zero verification only, not stochastic sampling.
"""


def verified_tokens(proposals, predictions, remaining, eos_ids=()):
    """Keep the matching draft prefix, then the target's correction/bonus token."""
    if remaining < 1 or len(predictions) != len(proposals) + 1:
        raise ValueError("Need a positive budget and one extra target prediction")
    accepted = 0
    while accepted < len(proposals) and proposals[accepted] == predictions[accepted]:
        accepted += 1
    emitted = (proposals[:accepted] + [predictions[accepted]])[:remaining]
    for index, token in enumerate(emitted):
        if token in eos_ids:
            emitted = emitted[:index + 1]
            break
    return emitted, min(accepted, len(emitted))


def check_inputs(prompt, max_new_tokens):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Expected one nonempty, unpadded prompt with shape (1, length)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")


def greedy_decode(target, prompt, max_new_tokens, eos_ids=()):
    """Optimized target-only baseline: one prefill, then one token per call."""
    import torch

    check_inputs(prompt, max_new_tokens)
    generated, cache, calls = [], None, 0
    inputs = prompt
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            output = target(input_ids=inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            inputs = output.logits[:, -1:].argmax(dim=-1)
            token = inputs.item()
            generated.append(token)
            calls += 1
            if token in eos_ids:
                break
    return {"token_ids": generated, "target_forward_calls": calls}


def speculative_decode(target, drafter, prompt, max_new_tokens, k=4, eos_ids=()):
    """Verify k proposals in one target call; discard rejected cached states."""
    import torch
    from transformers import DynamicCache

    check_inputs(prompt, max_new_tokens)
    if k < 1:
        raise ValueError("k must be positive")
    sequence = prompt
    generated, accepted, proposed, passes, target_calls, draft_calls = [], 0, 0, 0, 0, 0
    target_cache = DynamicCache(config=target.config)
    draft_cache = DynamicCache(config=drafter.config)
    with torch.inference_mode():
        if max_new_tokens and prompt.shape[1] > 1:
            # Leave the last context token pending so verification also yields a bonus token.
            target_cache = target(input_ids=prompt[:, :-1], past_key_values=target_cache,
                                  use_cache=True, logits_to_keep=1).past_key_values
            draft_cache = drafter(input_ids=prompt[:, :-1], past_key_values=draft_cache,
                                  use_cache=True, logits_to_keep=1).past_key_values
            target_calls += 1
            draft_calls += 1
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            count = min(k, remaining)
            draft_sequence = sequence
            for _ in range(count):
                # After full acceptance, consume both the last proposal and bonus token.
                inputs = draft_sequence[:, draft_cache.get_seq_length():]
                output = drafter(input_ids=inputs, past_key_values=draft_cache,
                                 use_cache=True, logits_to_keep=1)
                draft_cache = output.past_key_values
                next_token = output.logits[:, -1:].argmax(dim=-1)
                draft_sequence = torch.cat((draft_sequence, next_token), dim=-1)
                draft_calls += 1
            candidates = draft_sequence[:, sequence.shape[1]:]
            verify_input = torch.cat((sequence[:, -1:], candidates), dim=-1)
            output = target(input_ids=verify_input, past_key_values=target_cache, use_cache=True)
            target_cache = output.past_key_values
            predictions = output.logits.argmax(dim=-1)[0].tolist()
            emitted, kept = verified_tokens(candidates[0].tolist(), predictions, remaining, eos_ids)
            generated.extend(emitted)
            sequence = torch.cat((sequence, prompt.new_tensor([emitted])), dim=-1)
            # Both caches must exclude the new correction/bonus token before the next round.
            target_cache.crop(sequence.shape[1] - 1)
            draft_cache.crop(sequence.shape[1] - 1)
            accepted += kept
            proposed += count
            passes += 1
            target_calls += 1
            if emitted[-1] in eos_ids:
                break
    return {"token_ids": generated, "accepted_draft_tokens": accepted,
            "proposed_draft_tokens": proposed, "target_verification_passes": passes,
            "target_forward_calls": target_calls, "draft_forward_calls": draft_calls}
