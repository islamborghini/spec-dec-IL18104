# Project guide and current state

This is the working record of the project as of October 29, 2026. Historical
reports keep their original results; newer experiments do not overwrite them.

## Read this first

We are testing whether probability-distilling a small language model toward a
larger frozen model makes speculative decoding faster. We built the decoder,
prepared article-separated WikiText samples, trained LoRA adapters, and measured
generation. Training improved KL and acceptance, but speculative decoding still
loses to target-only generation in the measured setup. Profiling shows that
running the drafter is the dominant cost; our fixed k=4 also needs tuning.

The current assignment is the **demo/progress report**, not the final project
or debrief. A working pipeline and honest preliminary results satisfy the
technical milestone; a demonstrated speedup is not required. The one-page PDF
and linked screen recording have not been created.

## Documentation map

| Document | Purpose |
| --- | --- |
| [README](../README.md) | Installation, commands, data layout, checks |
| [Demo plan](DEMO_PLAN.md) | Submission requirements, recording outline, remaining work |
| [Baseline report](../reports/BASELINE.md) | Untrained speculative decoding and numerical controls |
| [Pilot report](../reports/TRAINING.md) | 500-chunk training, loss curves, generation comparison |
| [10k report](../reports/TRAINING_10K.md) | Same-run continuation and its evaluation |
| [Speed diagnosis](../reports/SPEED_DIAGNOSIS.md) | Paired checkpoints, k sweep, forward timing, limitations |
| [10,000-step continuation](../reports/TRAINING_TO_10K_STEPS.md) | Newly authorized fresh-data experiment and status |

Each experiment report links its raw JSON. Dataset manifests contain the full
revisions, checksums, sampling metadata, and recorded audit results.

## Code map

| File | Responsibility |
| --- | --- |
| [`modal_app.py`](../modal_app.py) | Workspace, image, secret, A10/W&B setup smoke check |
| [`prepare_data.py`](../prepare_data.py) | Article reconstruction, sampling, split/overlap checks, pilot artifacts |
| [`prepare_10k.py`](../prepare_10k.py) | Expanded dataset preserving the original pilot prefix/evaluation files |
| [`decoding.py`](../decoding.py) | Greedy target-only and cached speculative decoding |
| [`baseline.py`](../baseline.py) | Baseline GPU job, timing, pooled metrics, output checks |
| [`distillation.py`](../distillation.py) | Full-vocabulary KL and optimizer/RNG restoration |
| [`train.py`](../train.py) | LoRA training, validation/selection, checkpoint saving, generation comparison |
| [`continue_training.py`](../continue_training.py) | Guarded first continuation of the existing W&B run |
| [`prepare_fresh.py`](../prepare_fresh.py) | Fresh nonoverlapping windows and full official evaluation passage audit |
| [`continue_to_10k.py`](../continue_to_10k.py) | Restore at 2,625 and continue through 10,000 total updates |
| [`diagnose_speed.py`](../diagnose_speed.py) | Paired checkpoints, exploratory k sweep, forward timing |
| [`plot_training.py`](../plot_training.py) | CPU-only plots of saved training and validation history |
| [`tests/`](../tests/) | Dataset, decoder, distillation, setup, and continuation checks |

## Research question and comparisons

The target is **Qwen3-4B**. Its parameters stay frozen. The drafter is
**Qwen3-0.6B**, either its original weights or those weights plus trained LoRA
adapters. We compare three behaviors:

1. **Target-only:** the 4B model generates each next token itself. This is the
   reference for output equivalence and the baseline for acceleration claims.
2. **Original speculative:** the untrained 0.6B drafts tokens, which the target
   verifies. This isolates the value of training the drafter.
3. **Trained speculative:** the same pipeline uses a trained, merged drafter.

The 500-chunk and 10k checkpoints are also compared directly in the speed
diagnosis. Improving over the original drafter is different from beating the
target-only baseline. We have demonstrated the former, not the latter.

This implementation is **greedy/temperature-zero speculative decoding**. It
does not implement stochastic rejection sampling or claim an identical sampled
distribution. Numerical output equivalence must still be checked for greedy
generation.

## Models and fixed infrastructure

| Item | Actual setting |
| --- | --- |
| Target | `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Drafter | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| GPU | One NVIDIA A10, BF16 |
| Attention | Transformers SDPA |
| Remote Python | 3.12 |
| Core remote packages | torch 2.6.0; transformers 4.57.3; peft 0.17.1; wandb 0.31.0 |
| Modal workspace/profile | `islam-95550` |
| Persistent cache volume | `spec-dec-benchmark-model-cache`, mounted at `/cache` |
| W&B entity | `islamassanov66-minerva-university` |
| W&B project | `speculative-decoding` |
| Training W&B run | `hkays15c`, display name `a10-lora-kl-pilot` |

The A10 choice followed an earlier L4/A10 hardware check. Colab was considered
and an L4 allocation was confirmed, but the recorded project experiments use
Modal A10s. No claim here depends on unverified provider prices or guaranteed
Colab availability. Using the same GPU family reduces one source of variation;
it does not guarantee identical host hardware, clocks, numerical paths, or
throughput across allocations.

Model/tokenizer revisions and the shared tokenizer backend are pinned and
checked. GPU jobs require cached model weights instead of silently downloading
large checkpoints during timed work.

## Data preparation and leakage controls

Source: `Salesforce/wikitext`, `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. WikiText provides natural text for
controlled continuation experiments. It is a practical initial domain, not a
proven best dataset for speculative decoding or evidence about code/chat.

| Prepared artifact | Samples | Shape | Purpose |
| --- | ---: | --- | --- |
| `data/pilot/train.jsonl` | 500 | 256 input tokens | Initial training |
| `data/wikitext-10k/train.jsonl` | 10,000 | 256 input tokens | Expanded training |
| `data/pilot/validation.jsonl` | 50 | 256 input tokens | Loss, checkpoint selection, generation diagnostics |
| `data/pilot/test.jsonl` | 100 | 128 prompt + 128 reference tokens | Reserved final evaluation |

The 10k training file starts with the exact 500 pilot rows and adds 9,500 new
articles. The evaluation files remain byte-for-byte unchanged. The expanded
phase contains 2.56 million token positions; cumulative training exposure is
2.688 million positions because the initial 500 chunks were seen twice.

Preparation preserves the official source splits, reconstructs full articles,
shuffles articles with seed 42, and samples windows inside articles. Training
uses one window per distinct selected article. Validation has 50 articles;
100 test windows can share articles, with at most two per article.

Controls reject normalized title/article-hash matches across splits, duplicate
windows, and shared contiguous 32-token passages between selected splits.
Repeated source article titles are handled before choosing training windows.
The article-heading parser requires the expected surrounding blank lines so
equation-like text does not become a false article boundary. Manifests preserve
source rows, token offsets, hashes, versions, and rejected-sample counts.

These controls do not prove absence of semantic duplication or model-pretraining
contamination. Random sampling supplies variety, not immunity to overfitting.
Fixed validation measures progress; final testing remains reserved.

`input_ids` are authoritative. They have no padding, added special tokens, or
chat template. A prompt is a text extract to continue, not necessarily a question.
The saved Wikipedia continuation is not the target model's answer. Speculative
equivalence compares generated token IDs against target-only generation, not
against Wikipedia references or cosine similarity.

256-token training chunks are a budget/scope choice. They do not justify claims
about long-context generation. Current generation benchmarks use 128-token
prompts and up to 64 output tokens. Longer prompt/output lengths must be measured
before making broader speed claims.

## What training actually does

This is **output-probability distillation**, not activation distillation or a
separately trained prediction head. The frozen target and trainable drafter see
the same text prefixes. The objective is mean per-position full-vocabulary
`KL(target || drafter)`, with FP32 log-softmax and temperature one. We do not
train against teacher-written answers, hidden-state activations, or a text
similarity score.

| Setting | Value |
| --- | --- |
| Trainable weights | Rank-8 LoRA on `q_proj` and `v_proj` |
| LoRA alpha / dropout | 16 / 0 |
| Trainable parameter count | 1,146,880 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0 |
| Microbatch / gradient accumulation | 1 / 4 |
| Effective batch | 4 chunks, 1,024 token positions per update |
| Gradient clipping | Global norm 1.0 |
| Pilot | One pass over 500 chunks, 125 optimizer updates |
| Expanded phase | One pass over 10k chunks, 2,500 additional updates |
| Global update numbers | Pilot 1–125; continuation 126–2,625 |

**125 means optimizer updates, not 125 LoRA adapters.** Rank, trainable parameter
count, chunks, microbatches, and optimizer steps are distinct quantities.
These settings are practical pilot choices; no scientific claim says they are
optimal, and no learning-rate or LoRA-rank sweep has been performed.

Validation evaluates all 50 fixed chunks before training and periodically
during it: every 25 pilot updates, then every 250 continuation updates. Lowest
validation KL selects the checkpoint, including the phase-start checkpoint.
Both phases selected their last update. Best weights are restored and verified,
then merged into BF16 base weights to remove adapter inference overhead. Merge
effects on validation metrics are measured separately.

Training checks verify KL direction/normalization, frozen base/teacher weights,
adapter updates, and save/reload/merge behavior. Loss and gradient values must
remain finite. No additional epoch was launched after update 2,625.

## W&B logging and checkpoint continuation

[Training run and curves](https://forge.coreweave.com/wandb/islamassanov66-minerva-university/speculative-decoding/runs/hkays15c).

`train/kl`, `validation/kl`, and gradient metrics use `optimizer_step` as their
x-axis. W&B's internal `Step` counts logging events and can include validation
and metadata. The separate baseline run's approximately 20 timing events were
never a count of training updates.

Training-batch loss and gradient norm vary because each update sees different
articles. The gradient metric is the norm **before** clipping, not the learning
rate or Adam's weight-update magnitude. In the completed 10k phase, median norm
was 0.736, maximum 1.836, and 0.96% exceeded 1.0 before clipping. Fixed validation
KL declined throughout; no non-finite values were found. This is evidence
against a failed/exploding run, not proof of an optimal learning rate.

The user requested one continuous training graph. `continue_training.py` used
the same run ID with `resume="must"`, preserved original configuration, and
stored expanded settings under `continuation_from_step_125`.

The pilot did **not** save Adam moments. The first continuation restored adapter
weights but deliberately used a fresh optimizer; that is recorded in the report.
The expanded phase saved adapter weights plus optimizer/RNG state every 250
updates. Old pilot files were preserved in a separate directory.

| Artifact | Modal volume path below `/cache` |
| --- | --- |
| Pilot adapters | `project-runs/hkays15c/{best,last}/` |
| Expanded adapters | `project-runs/hkays15c/continuation-126-2625/{best,last}/` |
| Expanded optimizer/RNG state | `project-runs/hkays15c/continuation-126-2625/training_state.pt` |

Checksummed local copies are ignored by Git under
`results/adapters-hkays15c/`, with the expanded phase in `training-10k/`.
W&B also stores best/last model artifacts and evaluation artifact versions.

**The existing continuation entry point is specifically for the original
125→2,625 phase.** Its stale-history guard now refuses a rerun from update 125.
Saved current weights/optimizer enable further work, but an additional phase
needs a new checkpoint/data-order/step plan and loader. It is not implemented
as a generic arbitrary-step resume command. A separate specialized
`continue_to_10k.py` now implements the authorized 2,625→10,000 phase; see the
linked experiment plan for preparation and launch status.

## Decoding and metric definitions

The target-only loop prefills once and reuses its KV cache. The speculative
loop maintains separate target/drafter KV caches, drafts k tokens serially,
verifies the block in one target call, emits the accepted prefix plus a
correction/bonus token, and crops rejected cached states. EOS and output limits
are checked. Tiny real-Qwen tests cover rejection positions, full acceptance,
different k values, EOS, and budgets.

The fixed k=4 used for progress comparisons is an inference parameter, not a
training parameter requiring a separate adapter for every k. A validation sweep
can choose k after training; picking it using final test results would leak test
information into model selection.

| Metric | Meaning |
| --- | --- |
| KL | How closely next-token distributions match on fixed text prefixes |
| Teacher-forced top-one agreement | Fraction of fixed-prefix positions where argmax matches |
| Draft acceptance | Retained proposed tokens divided by all proposed tokens |
| Output tokens / verification | Generated tokens, including correction/bonus, per target verification |
| TPS | Total generated tokens divided by total synchronized elapsed seconds |
| Throughput ratio | Speculative TPS divided by target-only TPS in the same experiment |
| Sequence mismatch | Whether paired target/speculative output ID lists differ |
| Position mismatch | Differing token positions after divergence, not conditional token error probability |
| `verified_speedup` | Kept null when numerical output checks fail |

Timed generation includes prefill, drafting, verification, cache handling, and
Python overhead. It excludes model loading, tokenization, warm-up, and W&B
logging. Pooled TPS is not an unweighted average of per-prompt TPS.

## Results and what they support

The initial pilot reduced validation KL **1.06523→0.83503**. The 10k phase reduced
it further to **0.75725**. This is real progress in matching teacher probabilities.

The 10k training job measured **24.85 trained speculative TPS**, versus
**22.36 original speculative** and **32.87 target-only**. It improved over the
untrained drafter but did not accelerate target-only generation. Its 52.80%
draft acceptance should not be confused with 65.68% teacher-forced agreement.

The later same-job diagnosis compared both trained checkpoints directly:
**16.11 TPS pilot versus 16.77 TPS 10k**, a 4.13% gain. Its absolute TPS and
acceptance differ from the earlier job; numerical paths changed as well.
Use within-job controlled comparisons, not a subtraction of TPS across jobs.

Profiling assigned about **70%** of elapsed time to drafter decoding forwards,
versus **27%** to target verification. On a three-prompt subset, k=1 was faster
than k=4 but still slower than target-only. The benchmark also contains an
easy repeated-`=` continuation. The full diagnosis distinguishes these measured
findings from hypotheses about kernel-launch cost and training capacity.

## Numerical correctness remains unresolved

BF16 speculative outputs differ from target-only outputs in multiple cases.
An earlier diagnostic reproduced first differences at identical prefixes with
small/tied top logits, and a separate FP32-target control matched all five
original-drafter cases. That control was not repeated for every trained adapter,
k, context length, or serving implementation.

Changing floating-point calculation shapes can change near-tie argmax decisions,
after which whole continuations diverge. This does not justify ignoring the
differences. We record mismatches and make no verified lossless-acceleration
claim. Neither cosine similarity nor comparing to Wikipedia continuation tokens
would establish target equivalence.

## Operations, costs, and the crash-loop incident

W&B credentials are stored only in Modal Secret `wandb` as `WANDB_API_KEY`.
No keys belong in Markdown, source, logs, notebook cells, or commits.
All commands explicitly select `MODAL_PROFILE=islam-95550`; the account's
default profile may be different.

Apps are ephemeral. Successful jobs stop when their local entry point finishes;
this does not erase persistent checkpoints. At the end of the diagnosis,
the active-container list was empty. No inference service is deployed.

A temporary CPU-only, read-only W&B metrics probe initially failed because its
remote image lacked the `train` module. The replacement added the required
module and succeeded. The failed first app was mistakenly left active and
repeatedly failed startup until explicitly stopped. It did not train or create
a W&B run. The GPU training/evaluation job completed independently.

Important operational lesson: `retries=0` on function execution is not a
guarantee that failing container startup cannot repeat. Watch startup logs and
explicitly stop a failed app before launching a replacement. Prior setup also
encountered sparse W&B history rows and an unsupported Modal image override;
the committed continuation handles sparse rows and uses an explicit GPU wrapper.
Serialized functions require compatible local/remote Python versions; the
project uses source-based decorated functions instead.

Inspect activity without starting a GPU or W&B run:

```sh
MODAL_PROFILE=islam-95550 modal app list --json
MODAL_PROFILE=islam-95550 modal container list --json
```

To stop a failed app, use `modal app stop APP_ID --yes` with the explicit profile
and actual failed app ID. Confirm the active-container list afterward. Do not
stop a healthy training app merely because a separate diagnostic app failed.

The user reported $30/month in Modal credits. Current remaining credit and
actual billed totals were not checked. Recorded GPU function durations are
198.6 seconds for the pilot, 1,634.6 seconds for the 10k continuation, and 353.6
seconds for the speed diagnosis. These exclude some startup/upload/teardown and
are not invoices. Billing for the failed CPU probe is unverified. Timeouts bound
execution duration, not total dollar spend.

## Reproduction and next work

Follow the [README](../README.md) for installation and exact commands. Local
dataset audits and tests do not need GPU credentials. GPU commands allocate
resources; a new `train.py` run creates a new W&B run, while the specialized
continuation command intentionally refuses stale resumes.

The repository contains evidence sufficient to inspect the completed work
without rerunning it. `diagnose_speed.py` repeats the bounded diagnostic job
without creating a W&B run, but requires both saved adapter checkpoints in the
Modal cache. Its raw JSON and the production decoder source hashes are saved
with the committed diagnostic report.

Next priorities are the demo deliverables, explicit numerical controls, a larger
fixed validation benchmark, and lower draft latency/k tuning. Additional training
or a different model/serving backend should be a controlled experiment after
the timing bottleneck is addressed. Test prompts remain unused for generation.
