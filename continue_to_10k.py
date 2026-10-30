"""Resume Adam/LoRA at 2,625 and train fresh data through update 10,000."""

import hashlib
import json
from pathlib import Path

import modal

import train
from modal_app import ENTITY, PROJECT, WORKSPACE, wandb_secret

ROOT = Path(__file__).resolve().parent
RUN_ID = "hkays15c"
DATA = ROOT / "data/wikitext-fresh-to-10k-steps"
image = (train.image.add_local_python_source("train")
         .add_local_file(DATA / "train.jsonl", "/project/expanded/train.jsonl")
         .add_local_file(DATA / "manifest.json", "/project/expanded/manifest.json")
         .add_local_file(ROOT / "reports/training-10k.json", "/project/previous-training.json"))


def resume_plan(report, latest_step, manifest):
    if report["run_id"] != RUN_ID or latest_step != 2625 or report["train_history"][-1]["step"] != latest_step:
        raise ValueError("Expected matching saved weights and finished W&B history at update 2625")
    if manifest["train_chunks"] != 29500 or manifest["optimizer_updates"] != 7375:
        raise ValueError("Expected 29,500 fresh chunks for exactly 7,375 updates")
    if manifest["window_tokens"] != 256:
        raise ValueError("Expected 256-token chunks")
    if manifest["starting_optimizer_step"] != 2625 or manifest["ending_optimizer_step"] != 10000:
        raise ValueError("Wrong update interval")
    if manifest["prior_training_sha256"] != report["config"]["dataset_manifest"]["file_sha256"]["train.jsonl"]:
        raise ValueError("Fresh-data audit used a different prior training set")
    for split in ("validation", "test"):
        if manifest["file_sha256"][f"{split}.jsonl"] != report["config"]["dataset_manifest"]["file_sha256"][f"{split}.jsonl"]:
            raise ValueError("Frozen evaluation data changed")
    required_checks = {"overlap_with_prior_source_spans": 0, "duplicate_windows": 0,
                       "shared_32_token_passages_with_prior_training": 0,
                       "shared_32_token_passages_with_entire_official_evaluation": 0,
                       "shared_32_token_passages_between_new_windows": 0, "new_source_span_overlap": 0}
    if manifest["checks"] != required_checks or manifest["official_evaluation_articles_excluded"] != 120:
        raise ValueError("Fresh-data leakage audit did not pass")
    return {"run_id": RUN_ID, "start_step": 2625, "end_step": 10000, "train_chunks": 29500,
            "checkpoint_dir": report["storage_path"] + "/last",
            "checkpoint_sha256": {k.removeprefix("last/"): v for k, v in report["files"].items() if k.startswith("last/")},
            "optimizer_path": report["storage_path"] + "/training_state.pt",
            "optimizer_sha256": report["files"]["training_state.pt"],
            "optimizer_state_restored": True, "data_sha256": manifest["file_sha256"]["train.jsonl"],
            "job_limit_seconds": 7200,
            "note": "Resume adapter weights, Adam moments, and Torch/CUDA RNG; use only newly audited nonoverlapping chunks."}


@train.app.function(image=image, cpu=(1, 1), memory=(2048, 2048), secrets=[wandb_secret],
                    volumes={"/cache": train.cache}, timeout=120, startup_timeout=120, retries=0,
                    max_containers=1, scaledown_window=2, single_use_containers=True)
def preflight():
    import wandb

    run = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    if run.state == "running":
        raise RuntimeError("Training run is already running")
    history = run.scan_history(keys=["optimizer_step"], page_size=1000)
    latest = max((r["optimizer_step"] for r in history if r.get("optimizer_step") is not None), default=-1)
    report = json.loads(Path("/project/previous-training.json").read_text())
    manifest = json.loads(Path("/project/expanded/manifest.json").read_text())
    plan = resume_plan(report, latest, manifest)
    train.cache.reload()
    paths = {**{Path(plan["checkpoint_dir"]) / k: v for k, v in plan["checkpoint_sha256"].items()},
             Path(plan["optimizer_path"]): plan["optimizer_sha256"]}
    for path, expected in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return plan


@train.app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
                    volumes={"/cache": train.cache}, secrets=[wandb_secret], timeout=7200,
                    startup_timeout=120, retries=0, max_containers=1,
                    scaledown_window=2, single_use_containers=True)
def train_fresh_to_10k(revision, plan):
    return train.train_and_evaluate.local(revision, plan)


def main():
    import subprocess

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(f"Use MODAL_PROFILE={WORKSPACE}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the audited data and implementation before launch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    wandb_secret.hydrate()
    train.cache.hydrate()
    with modal.enable_output(), train.app.run():
        plan = preflight.remote()
        print(json.dumps(plan, indent=2), flush=True)
        print("CPU training checks:", train.check_training.remote(), flush=True)
        result = train_fresh_to_10k.remote(revision, plan)
    train.save_result(result, "training-to-10k-steps")


if __name__ == "__main__":
    main()
