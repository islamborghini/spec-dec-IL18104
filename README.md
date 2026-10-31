# Learning a drafter for speculative decoding

An IL18104 Deep Learning project testing whether probability distillation can
make a small language model a better drafter for speculative decoding.

A frozen **Qwen3-4B** target verifies proposals from **Qwen3-0.6B**. I train
LoRA adapters on the drafter to match the target's next-token distributions,
then compare the original and trained drafters using the same decoding code.

**Current result:** 10,000 optimizer updates improved validation alignment and
draft acceptance. Trained speculative decoding is faster than speculative
decoding with the original drafter, but remains slower than target-only
inference in the measured setup. Output mismatches remain unresolved, so this
project does **not** claim a verified lossless speedup.

[Training curves and artifacts in W&B](https://wandb.ai/islamassanov66-minerva-university/speculative-decoding/runs/hkays15c)
· [Detailed experiment report](reports/TRAINING_TO_10K_STEPS.md)
· [Recorded demo results](reports/DEMO_RECORDING.md)

## Try the recorded demo — no GPU required

From the repository root, run:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Open <http://127.0.0.1:8765/> and click **Start replay**. A wide desktop browser
shows four panels in a two-by-two grid; narrow windows stack them vertically.

| Panel | What runs in the original recording |
|---|---|
| Base drafter alone | Original Qwen3-0.6B generates independently, without verification |
| Target model alone | Qwen3-4B generates autoregressively; the speed baseline |
| Speculative, before training | Qwen3-4B verifies proposals from the original drafter |
| Speculative, after training | Qwen3-4B verifies proposals from the selected trained drafter |

The page replays **actual cached text and synchronized emission timestamps at
1× speed**, including speculative token bursts. Rates are measured averages
from that capture. The methods ran separately on one A10; their starts are
aligned in playback. This is recorded inference, not four live GPU jobs.

All methods received the same chat-formatted prompt with thinking disabled and
a 2,000-token output budget. The standalone models stopped at EOS before that
budget; speculative outputs reached the limit. Text is not padded or replaced.
This single prompt demonstrates behavior; it is separate from the WikiText
validation benchmark below. See [demo/README.md](demo/README.md) for rebuilding
the assets or making a new GPU recording.

## Results so far

The main comparison uses **20 fixed validation prompts, two timing repetitions,
128 prompt tokens, and up to 64 output tokens** on an NVIDIA A10, in BF16 with
SDPA. Both speculative methods use four draft tokens per verification (`k=4`).
Rows below come from the same evaluation job.

| Method | Tokens/sec ↑ | Retained draft acceptance ↑ | Output tokens / verification ↑ |
|---|---:|---:|---:|
| Target-only | **33.03** | — | — |
| Original drafter + target | 21.39 | 42.1% | 2.63 |
| Drafter at update 2,625 + target | 23.91 | 50.5% | 2.94 |
| Selected drafter at update 9,875 + target | **24.24** | **51.4%** | **2.97** |

On 50 fixed validation chunks, KL fell from **1.065 to 0.724**. Teacher-forced
top-1 agreement ended at approximately **66.8%**. The selected checkpoint is
update **9,875**, chosen by lowest validation KL; training completed update
10,000.

These results show improved imitation and acceptance, with diminishing gains
from later training. They do not establish that further training cannot help.
The final evaluation reports **22/40 sequence mismatches** between trained
speculative and target-only outputs; `verified_speedup` remains `null`.
[Raw results](reports/training-to-10k-steps.json) include token IDs, timings,
checkpoint hashes, and configuration.

A separate validation sweep of `k ∈ {1, 2, 4, 8}` found `k=1` fastest at
**29.15 tokens/sec**, versus **33.67 target-only** in that sweep's job. None of
the tested settings beat target-only in aggregate. Do not combine absolute
throughput across different jobs to estimate a training effect.
[Raw sweep](reports/k-sweep.json) · [Speed diagnosis](reports/SPEED_DIAGNOSIS.md)

## How it works

**Training:** the target stays frozen. On existing WikiText prefixes, the
drafter learns to minimize `KL(target || drafter)` over the full shared
next-token vocabulary, averaged over chunk positions. This is probability
distillation, not activation matching or training on generated answers.

**Inference:** the drafter proposes up to `k` tokens. The target verifies the
block in one forward pass. The decoder retains the matching prefix and emits
a target correction or bonus token, then adjusts the caches for the next
round. Both models maintain separate KV caches. The original and trained
drafters use the same inference path; trained adapters are merged before timing.

The implementation is **greedy, batch-one, unpadded decoding**. It does not
implement stochastic speculative rejection sampling, dynamic draft lengths,
quantization, or an optimized serving engine. Draft length is an inference
setting, not the number of training adapters.

## Data and training

Data comes from pinned **WikiText-103 raw** splits, tokenized with the shared
Qwen tokenizer. Training chunks contain **256 tokens** without a chat template.
Generation prompts are text extracts to continue, rather than questions.

| Training phase | Chunks in one pass | Added optimizer updates | Total updates |
|---|---:|---:|---:|
| Pilot | 500 | 125 | 125 |
| Expanded training | 10,000 | 2,500 | 2,625 |
| Fresh-data continuation | 29,500 | 7,375 | 10,000 |

The expanded phase includes the original 500 chunks. Overall there were
**40,000 chunk exposures, 39,500 unique chunks, and 10.24 million token
positions processed**. This is three staged passes over differently sized
datasets, not three epochs over one fixed dataset.

- **Optimizer:** AdamW, learning rate `1e-4`, zero weight decay, gradient clipping
  at 1.0; microbatch one with four accumulation steps per optimizer update.
- **LoRA:** rank 8, alpha 16, dropout 0, query/value projections only;
  1,146,880 trainable parameters.
- **Resume:** the first expansion restored adapter weights but reset Adam.
  The final continuation restored adapter weights, Adam state, and RNG state.
  Both continuations logged to the same W&B run, `hkays15c`.
- **Selection:** 50 fixed validation chunks monitor KL and choose checkpoints.
  The 100 frozen test prompts have not been used for generation evaluation.

Sampling is seeded and shuffled across articles. Manifests record source spans,
model/tokenizer revisions, seeds, checksums, and overlap audits. The final
29,500 chunks exclude previous training spans and shared 32-token passages, as
well as passages from all official evaluation articles. Validation/test files
remain unchanged. These checks do not rule out paraphrases or exposure during
Qwen's original pretraining.

Prepared data and audit details:
[data/pilot](data/pilot/manifest.json),
[data/wikitext-10k](data/wikitext-10k/manifest.json), and
[data/wikitext-fresh-to-10k-steps](data/wikitext-fresh-to-10k-steps/manifest.json).

## Inspect and reproduce

### Local setup and checks

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The local suite checks data provenance/leakage guards, resume safeguards,
decoder bookkeeping, and reconstruction of recorded output. Torch/PEFT-dependent
checks skip if those packages are absent locally; the training and baseline
runners execute their relevant checks on Modal CPU before GPU allocation.

The committed JSON reports and browser assets can be inspected without GPU
access or W&B credentials. Runtime training curves live in the linked W&B run;
`train/gradient_norm` is measured **before** clipping, and validation top-1
agreement uses fixed real-text prefixes rather than free-generation acceptance.

### GPU runs on Modal

The existing runners are configured for workspace **`islam-95550`**, W&B entity
**`islamassanov66-minerva-university`**, and project **`speculative-decoding`**.
For another account, update the configuration in `modal_app.py` and relevant
run/checkpoint references before running cloud jobs.

Authenticate without changing another default profile:

```sh
modal token new --profile islam-95550 --no-activate
```

Create a Modal secret named **`wandb`** containing **`WANDB_API_KEY`**, using an
account with access to the W&B entity. Keep credentials out of the repository.
Then the infrastructure smoke check can be run with:

```sh
MODAL_PROFILE=islam-95550 python modal_app.py
```

Expected: an A10 BF16 matrix check passes, a W&B setup run is recorded, and
hardware/check results are written to `results/setup.json`.

**Model runs additionally require the existing Modal volume
`spec-dec-benchmark-model-cache`, populated with the pinned model snapshots.**
The current model loaders use cached weights and fail if they are missing; a
fresh clone alone does not provision this cache. Continuations and trained
inference also require saved adapters/checkpoints in that volume. Weight files
are not included in Git; reports record their hashes and storage paths, and
training artifacts are linked through W&B.

| Entry point | Purpose / expected output |
|---|---|
| `prepare_data.py` | Rebuild the 500/50/100 pilot splits and manifest on CPU |
| `prepare_10k.py` | Rebuild the expanded 10,000-chunk training dataset |
| `prepare_fresh.py` | Rebuild the audited 29,500 fresh training chunks |
| `baseline.py` | Small five-prompt validation baseline → `results/baseline.json` |
| `train.py` | New 500-chunk pilot → `results/training.json` and adapters |
| `continue_training.py` | Historical update-125 → 2,625 continuation |
| `continue_to_10k.py` | Historical update-2,625 → 10,000 continuation |
| `diagnose_numerics.py` / `diagnose_speed.py` | Numerical checks / inference timing diagnosis |
| `k_sweep.py` | Selected-adapter validation sweep → `reports/k-sweep.json` |
| `record_demo.py` | New real four-method capture → `reports/demo-recording.json` |

Run GPU entry points as `MODAL_PROFILE=islam-95550 python <script>.py` after
checking their prerequisites. Data preparation uses CPU and may download the
pinned dataset/tokenizers; committed data is ready to inspect without rebuilding.
The continuation scripts are **phase-specific**, require matching saved state
and W&B history, and intentionally reject rerunning completed phases. They are
not generic commands to resume the current run from any step. Runners that
require a clean Git tree refuse uncommitted changes.

Cloud commands allocate paid resources; function timeouts are not dollar caps.
Reports under `reports/` preserve completed experiments, while most local
`results/` outputs are ignored by Git. Some diagnostic commands write directly
to report files; preserve existing measurements before rerunning them.

## Evaluation and limitations

Throughput is total generated tokens divided by total synchronized elapsed
time. Validation timing includes prefill and decoding, excludes loading,
tokenization, warmup, and W&B logging, and rotates method order. Acceptance is
retained draft tokens divided by proposed draft tokens; correction/bonus tokens
count toward output but not acceptance. Verification counts exclude prefill.

The key limitations are:

- **Numerical equivalence:** output mismatches remain. BF16 sensitivity is a
  candidate explanation, not proof that every difference is harmless.
- **Benchmark scope:** short Wikipedia continuations on one GPU family, batch
  size one, with only 20 validation prompts and repeated measurements. Repetitive
  outputs can inflate acceptance. This does not establish chat/code performance.
- **Inference cost:** an earlier instrumented subset spent about 70% of elapsed
  time inside drafter decode forwards. Kernel-level causes have not been isolated;
  a small parameter count alone does not guarantee a cheap decoding step.
- **Long outputs:** the browser capture is a single chat example with different
  output lengths, not a controlled long-context or long-generation benchmark.

Next steps are to resolve output-equivalence checks, reduce and measure drafter
inference cost with a fair target-only comparison, and extend validation across
context/output lengths. Final settings should be frozen before evaluating the
held-out test set. More training and different adapters remain possible
experiments; the current results do not establish their ultimate ceiling.

## Repository guide

| Location | Contents |
|---|---|
| `decoding.py` | Greedy baseline, speculative verification, KV-cache handling, emission callbacks |
| `distillation.py`, `train.py` | KL objective, LoRA training and evaluation |
| `prepare_*.py`, `data/` | Data preparation, prepared chunks and provenance manifests |
| `continue_*.py` | Audited training continuations and checkpoint restoration |
| `reports/` | Human-readable reports, raw measurements and saved figures |
| `demo/` | Offline recorded-output comparison in the browser |
| `tests/` | Local and remote preflight checks |
| `docs/PROJECT_GUIDE.md` | Detailed project history, infrastructure and design |
| `docs/DEMO_PLAN.md` | Progress-report/video submission plan |

Earlier measurements remain in [the baseline report](reports/BASELINE.md),
[pilot report](reports/TRAINING.md), and
[first expansion report](reports/TRAINING_10K.md). Their sample sizes and timing
conditions differ; use each report's own paired comparisons.

## Sources and data licensing

- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
  — speculative decoding background.
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
  and [dataset paper](https://arxiv.org/abs/1609.07843) — Wikipedia text source.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and
  [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) — pretrained models; exact
  revisions are pinned in the data manifests and experiment reports.

WikiText lists CC BY-SA 3.0 / GFDL licensing. Prepared samples retain article
provenance; applicable dataset and model licenses continue to apply.
