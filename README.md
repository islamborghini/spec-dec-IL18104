# Speculative decoding — IL18104

This project will test whether distilling a Qwen3-0.6B drafter toward a frozen
Qwen3-4B target improves speculative decoding acceptance and generation speed.

**Current stage: repository, Modal, and Weights & Biases setup only.** Data
preparation, decoding, training, and project evaluation are not implemented yet.
The setup smoke test checks real BF16 computation on an A10 and records hardware
measurements in W&B. It does not report training results.

## Install the local runner

Use Python 3.10 or newer. Only the Modal client is needed locally; PyTorch and
W&B are installed in the remote container at pinned versions.

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

## Local checks

```sh
python -m unittest discover -s tests -v
```

These checks require no API keys or GPU. They verify that the runner stops before
launching a cloud app when the workspace is wrong or the W&B secret is missing.
The remote smoke test above checks the actual GPU and online W&B connection.
