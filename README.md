# Speculative decoding — IL18104

This project will test whether distilling a Qwen3-0.6B drafter toward a frozen
Qwen3-4B target improves speculative decoding acceptance and generation speed.

**Current stage: infrastructure and a fixed WikiText pilot dataset.** Decoding,
training, and project evaluation are not implemented yet.
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
The remote smoke test above checks the actual GPU and online W&B connection.
