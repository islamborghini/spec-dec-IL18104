"""Continue the existing W&B run on 10k chunks; refuse stale history/checkpoints."""

import hashlib
import json
from pathlib import Path

import modal

import train
from modal_app import ENTITY, PROJECT, WORKSPACE, wandb_secret

RUN_ID = "hkays15c"
ROOT = Path(__file__).resolve().parent
image = (
    train.image.add_local_python_source("train")
    .add_local_file(ROOT / "data/wikitext-10k/train.jsonl", "/project/expanded/train.jsonl")
    .add_local_file(ROOT / "data/wikitext-10k/manifest.json", "/project/expanded/manifest.json")
    .add_local_file(ROOT / "reports/training.json", "/project/pilot-training.json")
)


def continuation_plan(report, latest_step, manifest):
    last = report["train_history"][-1]
    if report["run_id"] != RUN_ID or latest_step != last["step"]:
        raise ValueError("W&B history and saved pilot checkpoint must agree; refusing to restart stale weights")
    if manifest["train_chunks"] != 10000 or manifest["window_tokens"] != 256:
        raise ValueError("Expected the audited 10k × 256-token dataset")
    if manifest["file_sha256"]["validation.jsonl"] != report["config"]["dataset_manifest"]["file_sha256"]["validation.jsonl"]:
        raise ValueError("Validation data changed")
    return {"run_id": RUN_ID, "start_step": last["step"], "train_chunks": 10000,
            "end_step": last["step"] + 2500,
            "checkpoint_dir": report["storage_path"] + "/last",
            "checkpoint_sha256": {name.removeprefix("last/"): value for name, value in report["files"].items()
                                  if name.startswith("last/")},
            "data_sha256": manifest["file_sha256"]["train.jsonl"],
            "optimizer_state_restored": False,
            "note": "Continue adapter weights; Adam moments restart because pilot did not save optimizer state."}


def last_optimizer_step(history):
    return max((row["optimizer_step"] for row in history if row.get("optimizer_step") is not None), default=-1)


@train.app.function(image=image, cpu=(1, 1), memory=(2048, 2048),
                    secrets=[wandb_secret], volumes={"/cache": train.cache},
                    timeout=90, retries=0, max_containers=1,
                    scaledown_window=2, single_use_containers=True)
def inspect_continuation():
    import wandb

    # Read-only API: this preflight never calls wandb.init or creates a W&B run.
    run = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    if run.state == "running":
        raise RuntimeError("The W&B training run is already running")
    history = list(run.scan_history(keys=["optimizer_step"], page_size=500))
    latest = last_optimizer_step(history)
    report = json.loads(Path("/project/pilot-training.json").read_text())
    manifest = json.loads(Path("/project/expanded/manifest.json").read_text())
    plan = continuation_plan(report, latest, manifest)
    train.cache.reload()
    for name, expected in plan["checkpoint_sha256"].items():
        path = Path(plan["checkpoint_dir"]) / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved checkpoint checksum changed: {name}")
    return {**plan, "wandb_run_name": run.name, "wandb_run_state": run.state,
            "history_rows_with_optimizer_step": len(history)}


@train.app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
                    secrets=[wandb_secret], volumes={"/cache": train.cache},
                    timeout=3600, startup_timeout=300, retries=0, max_containers=1,
                    scaledown_window=2, single_use_containers=True)
def resume_training(code_revision, plan):
    return train.train_and_evaluate.local(code_revision, plan)


def main():
    import subprocess

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(f"Use MODAL_PROFILE={WORKSPACE}")
    wandb_secret.hydrate()
    train.cache.hydrate()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the implementation and audited dataset before launch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with modal.enable_output(), train.app.run():
        plan = inspect_continuation.remote()
        print(json.dumps(plan, indent=2), flush=True)
        print("CPU training checks:", train.check_training.remote(), flush=True)
        result = resume_training.remote(revision, plan)
    train.save_result(result, "training-10k")


if __name__ == "__main__":
    main()
