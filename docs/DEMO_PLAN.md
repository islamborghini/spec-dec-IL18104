# Demo/progress submission plan

This plan covers the demo assignment shown in the course instructions, not the
final project or project debrief. The screenshot lists the demo as 10% weight,
due October 29, closing November 1. Verify the course portal for submission
availability; this repository does not verify whether an upload was accepted.

## What the assignment asks for

At the demo stage, data → training → evaluation should run end to end. Small
models, small datasets, and short runs are acceptable. Evaluation must exist,
model behavior must be inspectable, and appropriate sanity checks should run.
Results do not have to beat the baseline. Report current evidence and the plan
for completing the project.

The requested submission is a **one-page PDF** containing:

- A short progress summary with a figure/screenshot, such as a loss curve or
  sample output.
- A completion plan, including any cuts and why.
- A GitHub repository link with a useful README, reproduction instructions,
  and expected outputs.
- A link to a **2–5-minute screen recording** demonstrating the current project,
  evaluation versus baseline, inspection of training/outputs, and lessons learned.

There is no commit-count requirement in the supplied demo instructions. Clear,
reproducible work matters; do not manufacture commit history.

## Current completion checklist

- [x] Modal A10 and W&B connection configured.
- [x] Fixed article-separated WikiText data and reproducible manifests prepared.
- [x] Target-only and original-drafter baselines implemented and evaluated.
- [x] LoRA probability-distillation pilot and expanded 10k phase completed.
- [x] Same W&B training run continued through update 2,625.
- [x] Training/validation KL, examples, metrics, checkpoints, and checks saved.
- [x] Same-job checkpoint comparison, exploratory k sweep, and latency diagnosis completed.
- [x] Repo README and Markdown experiment reports available.
- [ ] Write and visually verify the one-page progress PDF.
- [ ] Record the 2–5-minute demo and save a viewer-accessible link.
- [ ] Add the recording link to the PDF and check repo/video access.
- [ ] Submit through the course portal and verify receipt.

Neither the PDF, recording, nor portal submission has been completed by the
work documented here.

## Suggested one-page content

**Question:** Can distilling Qwen3-0.6B toward frozen Qwen3-4B improve speculative
decoding acceptance and generation latency?

**Implemented:** fixed WikiText data, cached greedy speculative decoding, LoRA
KL training, held-out validation, paired generation timings, and correctness
checks on a Modal A10. Include the repo and W&B links.

**Figure:** use [the combined loss figure](../reports/training_10k_loss.png).
Validation KL fell 1.065→0.835 in the pilot and to 0.757 after the 10k phase.

**Result:** in the 10k training job, trained speculative generation measured
24.85 tokens/sec, versus 22.36 original speculative and 32.87 target-only.
Acceptance improved, but there was no target-only acceleration. Label this
small validation benchmark, with five prompts and two timing repetitions.

**What we learned:** same-job diagnosis finds expensive drafter forwards and
poor k=4 latency. The extra 10k training phase yields limited additional speed.
BF16 differences and repetitive benchmark outputs require further controls.

**Next plan:** improve the evaluation/numerical controls, reduce drafter latency,
tune k on validation, then freeze settings before final testing. Defer broad
code/chat domains, large model changes, and extensive context sweeps beyond
what is needed for this demo.

Keep results from different jobs in separate labeled tables. The later diagnosis
has different absolute TPS; use its within-job comparisons to explain the
bottleneck rather than combining measurements into one synthetic result.

## Suggested 3–4-minute recording

| Approximate time | Show and explain |
| --- | --- |
| 0:00–0:30 | Research question, target/drafter, three comparison methods |
| 0:30–1:15 | W&B training run, actual optimizer-step axis, train and validation KL |
| 1:15–2:00 | Saved generation example and original/trained/target-only measurements |
| 2:00–2:45 | Drafter cost and k sweep; distinguish acceptance from wall-clock speed |
| 2:45–3:30 | Sanity checks, BF16/repetition limitations, and next steps |

Use saved results and curves. A live rerun of the full training is unnecessary
for the recording. Do not show API keys or secret values. Explain the work in
your own words so you can answer questions about it.

## What remains for the final project

The supplied final-project instructions separately request a public GitHub repo
with code/tests/README, a quick reproducible version, full-run costs, source-data
provenance, trained weights or links, and a published write-up connecting the
work to course concepts. The current repository was private at setup; a public
release has not been performed. Publication/access changes and a final write-up
are separate work, not implied by completing the demo pipeline.

The final write-up should support claims with curves, outputs, and failure cases;
negative results are acceptable. The later debrief assesses understanding and
future improvements. Prepare to explain why KL, teacher-forced agreement,
acceptance, tokens per verification, and TPS are different measurements.
