# Continuation to 10,000 total optimizer updates

Status: **complete** (all 10,000 updates, W&B run `hkays15c`, A10). Raw results:
[training-to-10k-steps.json](training-to-10k-steps.json).

## Results

First launch failed before W&B resume: the trainer read `training_state.pt`
from inside `last/`, but it is saved beside it. Fixed in `ec41655`; relaunch
restored Adam/RNG at update 2,625.

Validation KL (50 fixed chunks): 1.065 (untrained) → 0.835 (125) → 0.757
(2,625) → **0.724 best at 9,875**; 0.726 at 10,000. Flat after ~8,600.
Teacher-forced top-1 agreement ends at 66.8%.

Same job, 20 validation prompts × 2 repetitions, 64 tokens, k=4:

| Setup | Tokens/s | Acceptance | Tokens / target pass |
|---|---:|---:|---:|
| Target-only | **33.03** | — | 1.00 |
| Original drafter | 21.39 | 42.1% | 2.63 |
| Update-2,625 drafter | 23.91 | 50.5% | 2.94 |
| Update-9,875 drafter | 24.24 | 51.4% | 2.97 |

Paired per-prompt timing: the 9,875 drafter beats the 2,625 drafter on 26/40
pairs (median 1.02×; sign test p≈0.08, not significant), beats the original on
38/40 (median 1.12×), and loses to target-only on 38/40 (median 0.72×).

Reading: 3.8× more training bought ~1.4% throughput. Training is near its
ceiling for this setup; the gap to target-only is drafter cost and k=4 (see
[SPEED_DIAGNOSIS.md](SPEED_DIAGNOSIS.md)), not undertraining. BF16 mismatches
(22/40 sequences) still block any lossless-speedup claim. Test set untouched.

Prepared data: **29,500 chunks from 28,045 distinct training articles**, with
1,448 repeated-passage candidates rejected. All six leakage/overlap audit counts
are zero. Training-file SHA256:
`c69ac3db460bed5dd5aa340d1f4db2131245809a3d6ee3be8b5ad189ddb8f16f`.
Validation/test checksums remain unchanged. See the
[manifest](../data/wikitext-fresh-to-10k-steps/manifest.json).

## Fixed plan

- Resume W&B run `hkays15c` at global update 2,625, restoring last adapter,
  Adam moments, and Torch/CUDA RNG from checksummed saved files.
- Add **7,375 updates**, ending at **10,000 total optimizer updates**.
- Use **29,500 fresh chunks × 256 tokens**, effective batch four, one pass:
  **7.552 million additional token positions**, without recycling prior chunks.
- Keep A10/BF16/SDPA, rank-8 query/value LoRA, learning rate 1e-4, and k=4.
- Check all 50 fixed validation chunks every 250 new updates, including the
  starting checkpoint; select lowest validation KL. Save adapters, optimizer/RNG,
  and partial progress JSON at the same interval.
- At completion compare target-only, original drafter, previous update-2,625
  drafter, and selected new drafter on the **first 20 fixed validation prompts**:
  128 context tokens, up to 64 generated tokens, two rotated-order repetitions.
  Keep and flag previously measured repetitive prompts rather than removing them.
- Continue the same W&B graph. Do not evaluate final test generations or tune
  settings using test results.

More training supplies more evidence; 10,000 steps is not a universal minimum
or an automatic threshold for statistical significance.

## Fresh-data policy

[`prepare_fresh.py`](../prepare_fresh.py) uses seed `20261029` and the same pinned
WikiText-103/tokenizer revisions. It excludes every official validation/test
article by normalized title and article hash. It blocks 32-token passages from
the **full 120 official evaluation articles** and the frozen project evaluation
windows, not just the prompts used in generation smoke tests.

The existing 10k training file is the prior-exposure registry; its first 500
entries cover the initial pilot. New source intervals cannot overlap prior
intervals. Candidates must also avoid prior-training 32-token passages,
duplicate windows, and 32-token passages shared among newly selected chunks.
An independent audit repeats the checks before writing the dataset.

Because there are fewer training articles than requested chunks, articles may
supply multiple **disjoint new windows**. Random-offset tiling uses only unused
source gaps. Seeded shuffled articles, randomized article-balanced rounds, and
a final shuffle give reproducible variety; this is not uniform sampling over
all possible overlapping windows. Old chunks are never recycled.

The checks cover known project data and official WikiText evaluation data,
not unknown external benchmarks, semantic paraphrases, or base-model pretraining
exposure. Passage matching uses 128-bit BLAKE2 fingerprints; any collision
conservatively rejects a candidate.

Files live in `data/wikitext-fresh-to-10k-steps/`. Earlier data is preserved.
The new manifest records source spans, seed, prior-exposure checksum, file
checksums, sample counts, and leakage-audit results.

## Running and safeguards

```sh
TOKENIZERS_PARALLELISM=false python prepare_fresh.py
python -m unittest discover -s tests -v
MODAL_PROFILE=islam-95550 python continue_to_10k.py
```

Commit prepared data/code first. Read-only preflight requires W&B history at
exactly 2,625, matching adapter/optimizer hashes, unchanged validation/test
checksums, and all fresh-data audit checks passing. It refuses stale reruns.
Remote CPU tests verify the resumed next Adam update and restored RNG exactly.

The GPU job has one container, no function retries, a separate startup limit,
and a **two-hour execution limit**. A timeout is not a dollar spending cap;
watch startup logs and explicitly stop a failed app. The previous 2,500 updates
took about 27 minutes; this phase is expected to take roughly three times as
long plus its larger evaluation. This is an estimate, not a bill.

New volume directory:
`/cache/project-runs/hkays15c/continuation-2626-10000/`.
Completed local output will be `results/training-to-10k-steps.json`, with
checksummed files under `results/adapters-hkays15c/training-to-10k-steps/`.
Old results and checkpoints remain intact.

This is a specialized phase entry point, not a generic resume from arbitrary
intermediate checkpoints. BF16 equivalence and kernel-level profiling remain
separate unresolved work.
