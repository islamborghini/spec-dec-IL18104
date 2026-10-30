# Speculative decoding — IL18104

This project will test whether distilling a Qwen3-0.6B drafter toward a frozen
Qwen3-4B target improves speculative decoding acceptance and generation speed.

**Current stage: infrastructure, a fixed WikiText pilot, greedy baseline
evaluation, and a completed one-epoch LoRA training/comparison pilot.**
The setup smoke test checks real BF16 computation on an A10 and records hardware
measurements in W&B. It does not report training results.

## Install the local runner

Use Python 3.10 or newer. The local dependencies cover Modal and CPU-only data
preparation; PyTorch and W&B are installed in the remote container.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Authenticate the `islam-95550` Modal workspace without changing an existing
default profile:

```sh
modal token new --profile islam-95550 --no-activate
```

The workspace must be eligible for GPU usage. Modal may require a payment method
even when using included credits.

## Connect W&B

1. Get an API key from your [W&B account settings](https://wandb.ai/settings).
2. In the [Modal Secrets panel](https://modal.com/secrets/islam-95550/main), create
   a secret named **`wandb`** containing **`WANDB_API_KEY`** with that value.
3. Use a W&B account with write access to the
   **`islamassanov66-minerva-university`** entity.

The W&B project name is **`speculative-decoding`**. The first successful run
creates the project if it does not exist and the account has permission.
Credentials stay in Modal Secrets; never put an API key in source code, command
history, a notebook, or a Git commit.

## Run the setup smoke test

```sh
MODAL_PROFILE=islam-95550 python modal_app.py
```

Expected result:

- The local runner verifies the workspace and required secret before building
  the image or allocating a GPU.
- A remote A10 executes a small BF16 matrix multiplication and checks its output.
- A completed W&B run named `a10-setup-smoke`, with job type `setup`, records
  `gpu/total_gib`, `gpu/free_gib`, and `checks/cuda_matmul_passed=1`.
- The terminal prints the GPU name, measurements, and W&B run URL. The same data
  is saved locally to `results/setup.json` (ignored by Git).
- The ephemeral app stops when the command completes; no persistent deployment
  or model training is started.

The function has a two-minute execution timeout, no automatic retries, and one
container maximum. Image building and startup add time beyond the function's
execution; Modal charges for allocated resources. Treat a failed W&B login or
missing permission as a setup failure, not a successful offline run.

The A10 selection comes from a separate L4/A10 hardware benchmark using the
planned model pair. It does not establish speculative decoding speedup or model
quality; those still need to be measured in the project.

## Run the k=4 validation baseline

```sh
MODAL_PROFILE=islam-95550 python baseline.py
```

Commit changes before launching so W&B records the exact code revision. The
runner checks the workspace, W&B secret, and existing model-cache volume first.
It runs tiny random-Qwen decoder tests on a CPU before allocating an A10. No
test-set file is mounted or opened by this job.

The GPU job loads the pinned Qwen3-4B target and original Qwen3-0.6B drafter in
BF16 using SDPA, reusing the `spec-dec-benchmark-model-cache` volume from the
hardware benchmark. Missing cached weights fail rather than starting a download
on the GPU. Both models stay resident for all timings. It uses the first five
fixed validation articles, their first 128 sampled token IDs as prompts, and up
to 64 generated tokens per prompt, stopping on the target's EOS IDs.

`decoding.py` implements batch-one, unpadded **greedy decoding only**. The target
baseline uses a KV cache and one full-prompt prefill. Speculative decoding uses
separate KV caches, four draft proposals per block (fewer at the output limit),
one target verification call, an accepted prefix plus a correction/bonus token,
and cache rollback on rejection. It preserves the final-token budget and EOS.
Stochastic rejection sampling, quantization, and dynamic block sizes are outside
this pilot.

After checking the target loop against Transformers `generate()` on a real
prompt, the job warms up both paths. It records two timed repetitions of each
method on each prompt, alternating method order. Synchronized wall time includes
prefill, all drafting/verification calls, cache management, and Python overhead;
it excludes model loading, tokenization, W&B logging, and warm-up. This is a
custom batch-one baseline, not a comparison with an optimized serving engine.

W&B receives per-call TPS, a sample-output table, aggregate checks, acceptance,
generated tokens per target verification, and a JSON artifact containing token
IDs, timings, versions, hardware, and source provenance. The runner also saves
`results/baseline.json` locally. TPS is total generated tokens divided by total
timed seconds, not an average of individual TPS values. Acceptance is retained
draft tokens divided by all proposed tokens; the correction/bonus contributes
to generated tokens but not draft acceptance. Target verification counts exclude
the initial prefill, and total forward-call counts are recorded separately.

We report sequence and token-position mismatch rates. A nonzero mismatch gates
the report's `verified_speedup` field to `null` pending investigation; numerical
effects may explain differences, but a faster mismatching run is not presented
as a verified acceleration. Five articles and two repetitions are a smoke
benchmark, not a final uncertainty estimate or block-size selection.

The GPU function has a **600-second execution timeout**, no retries, one
container, and rapid scale-down. Startup and model loading are billable; the
timeout is not a dollar spending cap. The job performs no training or k sweep.

The first real run and its numerical follow-up are documented in
[reports/BASELINE.md](reports/BASELINE.md). The untrained k=4 drafter was slower
overall; BF16 output differences are reported explicitly. A separate FP32 target
check matched on all five validation cases. Raw JSON reports are committed so
the evidence can be inspected without a GPU or W&B access.

## Train the demo pilot

```sh
MODAL_PROFILE=islam-95550 python train.py
```

The CPU preflight checks KL direction/normalization, frozen base weights,
adapter updates, and adapter save/reload/merge before the GPU job. Commit code
before launch. This uses the same A10 family, pinned models, BF16, SDPA, and
prepared dataset as the baseline. Training is one shuffled pass over 500 chunks:
microbatch one, accumulate four, 125 AdamW updates, constant learning rate
`1e-4`, zero weight decay, gradient clipping at 1.0. LoRA uses rank 8, alpha 16,
no dropout, and `q_proj`/`v_proj` only. These are pilot settings, not tuned optima.

The frozen Qwen3-4B supplies next-token distributions on each text prefix.
We minimize `KL(target || drafter)` over the full shared vocabulary, averaging
over all 256 positions in each chunk, using FP32 log-softmax and temperature one.
This is probability distillation on existing text, without activation matching
or generated training answers. Only adapter weights receive gradients.

Evaluate all 50 validation chunks at updates 0, 25, 50, 75, 100, and 125; keep
the checkpoint with lowest validation KL, including the initial checkpoint if
training does not help. W&B records training/validation KL and validation
teacher-forced top-one agreement. This agreement uses fixed text prefixes and
is distinct from speculative acceptance during free generation.

Save best and final adapters to both W&B model artifacts and
`spec-dec-benchmark-model-cache:/project-runs/<wandb-run-id>/`. Best weights are
restored and checked before merging for inference. The merged model's validation
loss is measured again to expose any BF16 merge effects. The original drafter
is loaded separately; both inference drafters use plain Qwen modules without
adapter execution overhead. Compare target-only, original speculative, and
trained speculative decoding with the same five validation prompts, k=4,
64-token output limit, two repetitions, and rotating execution order. Report
numerical mismatches explicitly using the baseline's existing checks.

The runner saves `results/training.json` and downloads checksummed adapter files
to `results/adapters-<run-id>/`. W&B has loss curves, model artifacts, sample
continuations, and pooled comparison metrics. GPU execution is limited to ten
minutes, with one container and no retries; no held-out test file is mounted.
The best checkpoint is chosen by validation loss; block-size tuning remains for
later. Training runs and their weights are unique per W&B run ID.

For the submitted loss figure, copy the completed JSON to
`reports/training.json` and run `python plot_training.py`; it writes
`reports/training_loss.png` without using a GPU.

The completed pilot is documented in [reports/TRAINING.md](reports/TRAINING.md).
Validation KL fell from 1.065 to 0.835; draft acceptance rose from 44.1% to 51.7%.
In the paired validation pilot, trained speculative throughput was 24.4 tokens/s
versus original speculative 22.5 and target-only 33.6. BF16 discrepancies remain;
these results do not establish lossless acceleration. The held-out test prompts
remain unused. Raw results and the loss figure are committed; adapters are in
the linked W&B run and Modal volume.

## Expanded data and continuation in the same W&B run

The continuation is complete: validation KL **0.83503 → 0.75725** over updates
126–2,625 in the same W&B run. In the same-job generation comparison, trained
speculative throughput was **24.85 tokens/sec**, versus **22.36** with the
original drafter and **32.87** target-only. BF16 discrepancies remain.
See the [continuation report and KL figure](reports/TRAINING_10K.md).

```sh
TOKENIZERS_PARALLELISM=false python prepare_10k.py
MODAL_PROFILE=islam-95550 python continue_training.py
```

The prepared `data/wikitext-10k/train.jsonl` contains **10,000 chunks from 10,000
distinct training articles**, 256 tokens each: **2,560,000 tokens**. Its first
500 rows exactly match the pilot training data; the other 9,500 add new articles.
Validation/test files remain byte-for-byte unchanged under `data/pilot/`.
The manifest pins versions, seed, source positions, checksums, and overlap checks.
Sampling excludes official evaluation articles, repeated training titles/articles,
duplicate windows, and shared 32-token passages with the fixed evaluation windows.
The artifact audit runs in the local test suite. CPU preparation never trains a
model or evaluates the test set.

`continue_training.py` resumes **W&B run `hkays15c`** with `resume="must"`; it
does not create a new W&B run. A read-only CPU preflight verifies the current
W&B history and saved pilot checkpoint before any GPU allocation. If history
has already advanced beyond that checkpoint, it refuses to train from stale
weights. The pilot ended at update **125**; one new epoch over 10k chunks with
effective batch four adds 2,500 updates, logging **126–2,625** on the existing
`optimizer_step` axis. Original W&B history/configuration is preserved; expanded
settings are recorded under `continuation_from_step_125`.

This is a **weight continuation with a fresh Adam optimizer**: the pilot saved
LoRA weights, not Adam moments. That optimizer reset is recorded explicitly.
The continuation saves adapter and optimizer/RNG checkpoints every 250 updates,
checks all 50 validation chunks, and selects the best validation checkpoint.
The starting checkpoint is eligible. k remains four; the same five validation
generation prompts are used for the final comparison, and test data remains unused.

The continuation GPU function has a **one-hour execution timeout**, one
container, and no automatic retries. It writes new files under
`/cache/project-runs/hkays15c/continuation-126-2625/`, preserving the original
pilot checkpoint. Local outputs go to `results/training-10k.json` and
`results/adapters-hkays15c/training-10k/`. This command is specifically the first
continuation of the pilot; rerunning it after history advances fails the
preflight rather than duplicating updates.

In the same W&B training run, `train/kl` and `validation/kl` use
`optimizer_step` as their x-axis. `train/gradient_norm` records the norm returned
by PyTorch **before** clipping to 1.0; it is not the learning rate or the size of
Adam's weight update. Training batches differ, so inspect the fixed validation
curve alongside the noisier batch metrics.

To export both training phases after the continuation finishes, copy
`results/training-10k.json` to `reports/training-10k.json` and run
`python plot_training.py --continuation`. This writes
`reports/training_10k_loss.png`, preserving the original pilot figure.

## Fixed WikiText pilot

The prepared files are committed under `data/pilot/`, ready to use without a
Hugging Face download. Reproduce them locally with:

```sh
TOKENIZERS_PARALLELISM=false python prepare_data.py
```

This uses CPU only, downloads the raw WikiText-103 dataset and the two tokenizer
files (no model weights), and refuses to overwrite a different existing dataset.
The dataset revision, Qwen tokenizer revisions, seeds, source row ranges, article
hashes, token offsets, package versions, and output checksums are saved in
`data/pilot/manifest.json`. Running again with the same environment must produce
identical files.

| File | Samples | Tokens per sample | Use |
| --- | ---: | --- | --- |
| `train.jsonl` | 500 | 256 input tokens | Distillation training; 128,000 tokens |
| `validation.jsonl` | 50 | 256 input tokens | Monitor loss and select a checkpoint |
| `test.jsonl` | 100 | 128 prompt + 128 reference tokens | Frozen final comparison |

HF rows are reconstructed into articles at level-one headings. We preserve the
official splits, exclude training articles matching any validation/test title or
normalized article hash, and sample across shuffled articles rather than taking
the beginning of the corpus. Train/validation each use one random window per
selected article. Test sampling takes one window per eligible article before a
second round, with at most two windows per article, sampled in separate halves
when there is room for two. All selected regions stay inside their article.

Checks reject duplicate windows, cross-split article matches, and any shared
contiguous 32-token passage between sampled splits. Test filtering gives
validation priority; training filtering gives both evaluation sets priority.
These checks do not detect semantic paraphrases or guarantee that Qwen never
saw the source text during pretraining.

`input_ids` are authoritative, unpadded token sequences without a chat template
or added special tokens; `text` is their human-readable decoding. Both model
tokenizer backends are checked for equality. Test `reference_ids`/
`reference_text` are the actual next Wikipedia tokens, reserved alongside the
prompt; they are **not** a teacher-generated answer and should not be used as the
expected output in the speculative-decoding equivalence check. That check must
compare against target-only generation with the same prompt and decoding setup.
Here a prompt is a Wikipedia text extract for continuation, not a question.
Claims from this dataset concern text continuation rather than question answering.

Shuffle training order during training; keep dataset membership fixed. Use
validation for decisions and test only after freezing the experiment, with the
same prompts for all baselines and the trained drafter. Test prompts can share
an article, so uncertainty estimates should group by article. Pilot sizes are a
demo budget, not a proven optimum, and article-balanced Wikipedia samples do
not establish performance on code or chat.

Sources: [WikiText dataset and license](https://huggingface.co/datasets/Salesforce/wikitext)
(CC BY-SA 3.0 / GFDL) and the
[original paper](https://arxiv.org/abs/1609.07843). Prepared samples retain their
source article titles and dataset provenance; applicable source licenses apply.

## Local checks

```sh
python -m unittest discover -s tests -v
```

These checks require no API keys, downloads, or GPU. They exercise article
boundaries, reproducibility, leakage guards, and the committed dataset checksums,
and verify that the runner stops before launching a cloud app when the workspace
is wrong or the W&B secret is missing.
Four tiny-Qwen cache tests are skipped locally when PyTorch is absent; the
baseline runner executes them on Modal CPU and requires all decoder checks to
pass without skips before starting a GPU. Decoder caching follows the pinned
[Transformers cache API](https://huggingface.co/docs/transformers/v4.57.3/en/kv_cache).
Three training checks are also skipped when local PyTorch/PEFT are absent;
`train.py` executes those on Modal CPU before allocating its GPU.
The remote smoke test above checks the actual GPU and online W&B connection.
