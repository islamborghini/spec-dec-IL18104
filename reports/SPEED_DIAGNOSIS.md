# Why speculative decoding is still slower

The main measured bottleneck is the cost of running the drafter. Additional
training improves agreement but does not reduce the cost of its forward passes.
Our fixed k=4 is also a poor latency setting on the diagnostic subset. The
generation benchmark is too small and includes a degenerate continuation, so
none of these measurements establishes final-project performance.

[Raw diagnostic records](speed-diagnosis.json) ·
[Diagnostic implementation](../diagnose_speed.py) ·
[Completed Modal app](https://modal.com/apps/islam-95550/main/ap-zJfwRW9OCz4Dyer7ncQKI9).

## What was tested

One bounded A10 job used the existing cached, pinned BF16/SDPA models and the
unchanged validation data. It performed no training, never opened the test file,
and did not initialize or create a W&B run. All adapters were merged before
inference, so adapter execution overhead cannot explain the comparison.

1. Target-only, original drafter, 500-chunk pilot adapter, and 10k adapter:
   five validation prompts, 128 prompt tokens, 64 output tokens, three repeated
   timings, rotating method order: 60 generations.
2. Trained 10k drafter with k=1, 2, 4, 8: the first three validation prompts,
   two timing repetitions, rotating k order: 24 generations.
3. Three separately instrumented k=4 generations with CUDA events bracketing
   target and drafter forward calls. These are excluded from the timing tables.

The job completed in 353.6 seconds before final teardown. The returned records
were checked for finite positive timings, expected sample counts, matching
checkpoint hashes, and agreement between profiled and decoder forward-call
counts. Modal's container listing was empty after completion.

## Same-job checkpoint comparison

| Method | Tokens/sec | Retained draft acceptance | Output tokens / verification |
| --- | ---: | ---: | ---: |
| Target-only | 25.08 | — | — |
| Original speculative, k=4 | 14.35 | 34.57% | 2.319 |
| 500-chunk pilot speculative, k=4 | 16.11 | 41.75% | 2.602 |
| 10k speculative, k=4 | 16.77 | 44.66% | 2.712 |

The 10k adapter was **4.13% faster than the 500-chunk pilot**, and 16.87% faster
than the original drafter, in this job. The three 10k/pilot throughput ratios
were 1.0434, 1.0424, and 1.0381. This is more consistent than a single lucky
timing, but repetitions of the same five prompts are not independent new data
and do not establish broad statistical significance.

The earlier training job reported 22.36 original, 24.85 trained, and 32.87
target-only tokens/sec. Those numbers are preserved in [the training report](TRAINING_10K.md).
Absolute throughput changed across jobs, and BF16 generation paths also changed:
one of five target-only sequences and several speculative sequences differ
between the old and diagnostic jobs. The cause of that cross-job numerical and
latency variation was not isolated. Do not combine jobs to estimate a causal
training gain, or substitute this table for the original run's measurements.

Within the diagnostic job, original, pilot, and 10k speculative outputs were
identical in all 15 paired cases. That makes this comparison between drafters
easier to interpret. All three differed from target-only in 12/15 sequences.

## Measured bottleneck: the drafter

Aggregate of the three instrumented k=4 generations:

| Component | Stream elapsed time | Calls | Share of total wall time |
| --- | ---: | ---: | ---: |
| Drafter decoding forwards | 8,707.3 ms | 298 | 70.37% |
| Target verification forwards | 3,399.7 ms | 77 | 27.48% |
| Both model prefills | 210.9 ms | 6 | 1.70% |
| Residual outside forward-event spans | 55.8 ms | — | 0.45% |
| Total synchronized wall time | 12,373.7 ms | — | 100% |

A drafter decoding forward averaged **29.2 ms**. A target verification forward
averaged **44.2 ms**. With approximately four serial draft forwards per round,
the round averaged **160.7 ms** while producing roughly **2.49 output tokens**.
That is about 64 ms per output token; target-only generation on these three
prompts averaged about 40 ms per token. The saved inference time does not cover
the drafting cost.

CUDA events measure elapsed time on the stream, including possible host launch
gaps inside each forward. These are not a sum of pure GPU kernel runtimes.
The small residual does not rule out Python/kernel-launch overhead *inside*
model forwards. A kernel trace is needed to distinguish launch latency,
memory bandwidth, and arithmetic bottlenecks. It does show that cache rollback
and verification bookkeeping outside forwards are not the dominant measured
cost in these cases.

The parameter ratio is misleading for latency: the drafter still has
[28 transformer layers](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json),
versus [36 in the target](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json).
Both use ordinary eager PyTorch forward dispatch with SDPA. The layer counts
are consistent with substantial small-model dispatch cost, but are not by
themselves a measured explanation of every millisecond.

The [original speculative-decoding paper](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)
explicitly models draft-to-target latency cost, which depends on software and
hardware, rather than parameter count alone. Speed requires enough accepted
output to repay both drafting and verification.

## k=4 is not the best measured setting

These numbers use only the first three validation prompts; compare rows within
this table, not against the five-prompt averages above.

| Draft tokens per verification | Tokens/sec | Output tokens / verification |
| --- | ---: | ---: |
| 1 | 21.52 | 1.699 |
| 2 | 19.26 | 2.043 |
| 4 | 15.51 | 2.494 |
| 8 | 10.51 | 2.783 |

k=1 was **38.7% faster than k=4** on this subset, despite producing fewer tokens
per verification. Drafting more tokens costs more than the additional accepted
output saves. Target-only on the same three prompts averaged 25.04 tokens/sec,
so k=1 still did not beat it.

All six generations for each k differed from their paired target-only output.
These are exploratory timings with short warm-up and only two repetitions per
k; different k values can also change BF16 numerical paths. No k has been frozen
for final testing, and this is not a verified lossless speedup.

## Why more training had diminishing returns

In the original training evaluations, the 500-chunk pilot produced 2.963 tokens
per verification, versus 2.991 for the 10k checkpoint: less than 1% improvement
in that metric. In the new diagnostic job the corresponding improvement was
2.602 to 2.712, about 4.2%. Both show limited additional useful output, though
their absolute values differ because generation paths changed.

The objective minimizes full-distribution KL on fixed Wikipedia prefixes.
Our greedy verifier accepts a token only when the draft and target argmax agree
through the entire preceding proposed prefix. Lower KL need not change the
argmax or lengthen accepted prefixes. Training KL, teacher-forced top-one
agreement, and actual acceptance therefore answer different questions.

The 10k phase added 2.56 million token positions with rank-8 query/value LoRA;
it did not retrain the entire drafter. Capacity limits, the choice of modules,
and training on ground-truth rather than generated prefixes are plausible
additional limits, but no ablation identifies which is responsible. The
gradient checks and falling validation KL provide no evidence that the learning
rate caused a failed training run.

## A biased five-prompt smoke benchmark

The `Battle of Edson 's Ridge` case generates 64 repeated `=` tokens. Both models
agree perfectly in the original training comparison. That easy repetitive case
inflates average acceptance and throughput. Other sampled outputs also contain
repetition or instruction-like text inside raw completion.

As a labeled sensitivity check, excluding only that case from the *original*
10k training evaluation lowers trained acceptance from 52.80% to 45.96% and
throughput from 24.85 to 22.80 tokens/sec. Target-only becomes 32.98 tokens/sec.
This post-hoc exclusion is not a replacement benchmark or a headline result.
Keep all original records and report degeneracy explicitly. A larger validation
benchmark should be specified before evaluating it, with article-grouped
uncertainty and a stated output-quality/degeneracy policy.

## What was ruled out, and what should happen next

The code reuses separate KV caches; it does not forward the entire growing
prefix every round. Profiled decode inputs remain small. The target verifies a
whole proposal block in one call, and inference uses merged adapters. These
common accidental inefficiencies do not explain the observed cost.

Priority order:

1. Make output equivalence and BF16 numerical controls explicit, then establish
   a larger fixed validation benchmark. Keep the test set closed.
2. Profile kernels/launch gaps and reduce drafter latency, or evaluate a shallower
   compatible drafter. Apply comparable serving optimizations to target-only.
3. Reassess k on validation; k=1 is a promising diagnostic setting, not a final
   validated choice. Longer generations and contexts still need measurement.
4. Only then compare training changes such as on-policy teacher continuations,
   different LoRA modules/rank, or more tokens using controlled ablations.

No production decoder change, additional training epoch, or test-set evaluation
was performed for this diagnosis.
