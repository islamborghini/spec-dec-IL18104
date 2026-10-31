# Learning a drafter for speculative decoding

Does training a small drafter to imitate a larger target improve speculative
decoding throughput on one GPU? This IL18104 project uses **Qwen3-0.6B** as the
drafter and a frozen **Qwen3-4B** as the target, with WikiText-103 preparation,
LoRA probability distillation, validation, and generation benchmarks on an
**NVIDIA A10**.

## Results

Training completed **10,000 optimizer updates**, minimizing
`KL(target || drafter)`. Validation KL fell from **1.065 to 0.724**; the best
checkpoint was update **9,875**.

Same-job evaluation: 20 validation prompts, two timing repetitions, 128 prompt
tokens, up to 64 output tokens, BF16/SDPA, and four draft tokens per verification.

| Method | Tokens/sec | Draft acceptance |
|---|---:|---:|
| Target-only | **33.03** | — |
| Speculative, original drafter | 21.39 | 42.1% |
| Speculative, trained drafter | 24.24 | 51.4% |

**Training improved acceptance and speculative throughput, but did not beat
target-only generation.** A separate draft-length sweep found `k=1` fastest
at 29.15 tokens/sec, versus 33.67 target-only in that job. Output mismatches
remain unresolved, so no verified lossless speedup is claimed. The held-out
test set has not been used for generation evaluation.

[Full results](reports/TRAINING_TO_10K_STEPS.md) ·
[Raw measurements](reports/training-to-10k-steps.json) ·
[Speed diagnosis and k sweep](reports/SPEED_DIAGNOSIS.md) ·
[W&B training run](https://wandb.ai/islamassanov66-minerva-university/speculative-decoding/runs/hkays15c)

## Run the recorded demo

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Open <http://127.0.0.1:8765/> and click **Start replay**. Four panels compare
the small model alone, target alone, and speculative decoding before and after
training. No GPU or credentials are needed for playback.

The demo replays **real cached outputs and emission timestamps at 1×**.
Methods were recorded separately on one A10, with starts aligned for playback.
The shared prompt had a 2,000-token budget; EOS can stop outputs earlier.
This single chat example is separate from the validation benchmark above.
[Recording details](demo/README.md) · [Capture results](reports/DEMO_RECORDING.md)

## Setup and checks

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Local checks cover data audits, resume safeguards, decoder bookkeeping, and
replay reconstruction. Torch/PEFT-dependent tests skip when those packages
are absent locally; relevant checks run on Modal CPU before training/baseline
GPU jobs.

GPU runners are configured for Modal workspace `islam-95550`, W&B entity
`islamassanov66-minerva-university`, and project `speculative-decoding`.
They require Modal authentication, a `wandb` secret containing `WANDB_API_KEY`,
and pinned models in the existing `spec-dec-benchmark-model-cache` volume.
Trained inference also requires the saved adapter. A fresh clone alone does
not provision those weights; cloud runs allocate paid resources.

For example, with that setup in place:

```sh
MODAL_PROFILE=islam-95550 python baseline.py  # writes results/baseline.json
MODAL_PROFILE=islam-95550 python train.py     # new 500-chunk pilot, not the full run
```

The continuation scripts are phase-specific and reject stale reruns of the
completed training run. See the [project guide](docs/PROJECT_GUIDE.md) and
[10,000-update report](reports/TRAINING_TO_10K_STEPS.md) for prerequisites,
checkpoint paths, exact training settings, and expected outputs.

## Code and data

- `prepare_*.py`, `data/`: seeded WikiText sampling, prepared chunks, and leakage audits.
- `distillation.py`, `train.py`, `continue_*.py`: KL distillation, LoRA, and checkpoint resume.
- `decoding.py`, `baseline.py`, `k_sweep.py`: greedy KV-cached decoding and evaluation.
- `reports/`, `demo/`, `tests/`: saved evidence, browser replay, and checks.

Training used 256-token chunks, four chunks per optimizer update, rank-8
query/value LoRA, and learning rate `1e-4`. Across three phases, it processed
40,000 chunk exposures (39,500 unique chunks). Validation/test membership
stayed fixed; the last 29,500 chunks excluded previous training spans and
known evaluation passages. These audits do not rule out base-model pretraining
exposure or semantic overlap.

Next steps are to investigate output mismatches and drafter inference cost,
then freeze final settings before evaluating the held-out test set.

## References

[Speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html) ·
[WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext) ·
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) ·
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

Exact model/data revisions are pinned in the manifests and reports. Prepared
WikiText samples retain article provenance; applicable dataset licenses
(CC BY-SA 3.0 / GFDL) and model licenses apply.
