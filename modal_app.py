"""Verify the project's Modal A10 and W&B connection without training a model."""

import json
from pathlib import Path

import modal

WORKSPACE = "islam-95550"
ENTITY = "islamassanov66-minerva-university"
PROJECT = "speculative-decoding"

app = modal.App("spec-dec-il18104")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0", "wandb==0.31.0",
)
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])


@app.function(
    image=image, gpu="A10", cpu=(1, 1), memory=(4096, 4096),
    secrets=[wandb_secret], timeout=120, startup_timeout=300,
    retries=0, max_containers=1, scaledown_window=2,
    single_use_containers=True,
)
def check_setup():
    import torch
    import wandb

    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.is_bf16_supported(), "The project requires native BF16 support"
    gpu_name = torch.cuda.get_device_name(0)
    assert "A10" in gpu_name, f"Expected A10, received {gpu_name}"

    matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    torch.testing.assert_close(product, torch.full_like(product, 32))
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    metrics = {
        "gpu/total_gib": total_bytes / 1024**3,
        "gpu/free_gib": free_bytes / 1024**3,
        "checks/cuda_matmul_passed": 1,
    }

    with wandb.init(
        entity=ENTITY, project=PROJECT, job_type="setup", name="a10-setup-smoke",
        mode="online", settings=wandb.Settings(init_timeout=30),
        config={
            "gpu": gpu_name, "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "dtype": "bfloat16",
        },
    ) as run:
        run.log(metrics)
        result = {"gpu": gpu_name, "metrics": metrics, "wandb_url": run.url}

    return result


def main():
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(
            f"Expected Modal workspace {WORKSPACE!r}, got {workspace.name!r}. "
            f"Run with MODAL_PROFILE={WORKSPACE}."
        )
    # Check credentials before building an image or allocating a GPU.
    wandb_secret.hydrate()
    with modal.enable_output(), app.run():
        result = check_setup.remote()
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    (output / "setup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
