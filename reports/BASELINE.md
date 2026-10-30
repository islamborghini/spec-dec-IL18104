# Untrained drafter validation pilot

This is an initial negative result, not a final speedup claim. The untrained
Qwen3-0.6B drafter at k=4 was slower overall than target-only Qwen3-4B on these
five validation articles. No training, block-size selection, or test-set
evaluation has been performed.

[W&B run and artifacts](https://forge.coreweave.com/wandb/islamassanov66-minerva-university/speculative-decoding/runs/37tl67ah)
and [Modal execution](https://modal.com/apps/islam-95550/main/ap-9vH71KRE98sRWXcNSpVwYV).
The measured implementation is commit
`ca7bb7f906e27d1c6b319cca68f68b872e923a54`.

## Timing result

NVIDIA A10; BF16; SDPA; batch one; 128-token prompts; up to 64 output tokens;
five fixed validation articles; two repetitions with alternating method order.
All runs produced 64 tokens. Synchronized wall time includes prefill and all
decode work, and excludes loading, warm-up, tokenization, and logging.

| Metric | Result |
| --- | ---: |
| Target-only throughput | 33.40 tokens/sec |
| Speculative throughput, k=4 | 22.04 tokens/sec |
| Raw throughput ratio | 0.660× |
| Repetition ratios | 0.658×, 0.662× |
| Retained draft tokens / proposed draft tokens | 408 / 926 = 44.06% |
| Generated tokens per target verification | 2.689 |
| Accepted draft tokens per target verification | 1.714 |
| Peak allocated GPU memory during measurements | 8.66 GiB |

The generated-token metric includes corrections and bonus tokens. It excludes
the initial target prefill from its verification-pass denominator; total target
forward counts are also retained in the raw records. Resource estimates are not
actual bills and exclude startup, CPU checks, and failed attempts.

## Correctness and numerical follow-up

The tiny random-Qwen CPU tests passed for multiple block sizes, rejection,
full acceptance, cache rollback, EOS, and output budgets. The real target-only
loop also matched Transformers `generate()` on its 16-token calibration case.

However, **8 of 10 BF16 paired sequences differed**. There were 172 differing
positions out of 640 (26.875%); this is a positional sequence discrepancy,
not a conditional next-token error rate or task accuracy. One early token
change can cause many subsequent differences. The raw timing ratio therefore
has `verified_speedup: null` and must not be presented as verified acceleration.

We replayed the four affected articles. Their first divergences were at output
positions 25, 58, 27, and 23 (zero-based). Each trace confirmed an identical input
prefix at the divergence. Single-token and block-forward BF16 logits created
or broke top-candidate ties; top-two margins were 0 or 0.125 at these cases.
For example, one path gave logits 19.125 versus 19.0, while block verification
rounded both candidates to 19.125. The argmax tie-break can change the token.

Changing only target computation to FP32, while keeping the drafter in BF16,
produced exact target/speculative agreement on all five 64-token validation
cases. These were separate, untimed correctness checks; FP32 agreement does
not certify BF16 token identity or change the BF16 throughput numbers. The
evidence supports numerical precision as the cause of the observed first
divergences; it is not a proof for every possible prompt.

Raw evidence: [baseline.json](baseline.json), [numerics.json](numerics.json).

## Reproduction and next experiment

Run `MODAL_PROFILE=islam-95550 python baseline.py` from a clean committed checkout.
It checks the decoder on Modal CPU before starting the A10. To replay numerical
differences from the newly written local result, run
`MODAL_PROFILE=islam-95550 python diagnose_numerics.py`. The diagnostic uses a
separate A10 job with a five-minute execution timeout, then appends its evidence
to that baseline's W&B run using a CPU function.

Next: train one LoRA adapter using training chunks and validation distillation
loss; keep k=4 for progress comparisons. Tune block size on validation only
afterward. The held-out test prompts remain unused. Five articles and two
repetitions do not establish general performance or a statistically reliable
best configuration.
