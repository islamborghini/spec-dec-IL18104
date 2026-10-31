# Recorded four-method demo

Recording `demo-dcffd7998341` at 2026-10-31T03:41:32.159701+00:00; source code `876b75c2ecc980b135c52cd47773385a959a7131`.

Actual outputs and synchronized emission timestamps are saved in [demo-recording.json](demo-recording.json). The browser replays them at 1×; no authored substitute or constant-rate animation is used.

| Method | Tokens | Seconds | Tokens/sec | Stopping |
|---|---:|---:|---:|---|
| base | 1552 | 35.24 | 44.04 | eos |
| target | 1733 | 53.90 | 32.15 | eos |
| original | 2000 | 78.20 | 25.58 | token_limit |
| trained | 2000 | 77.38 | 25.85 | token_limit |

GPU: NVIDIA A10; bfloat16; SDPA. Shared chat-formatted prompt: 124 tokens; thinking disabled; output budget 2,000 per method. Both speculative paths use k=4; trained adapter selected at update 9875.

**Interpretation:** This is one new demonstration prompt, not the held-out test or a repeated benchmark. Different output lengths/content affect timing comparisons. Use the existing WikiText validation report for aggregate experimental results.

methods recorded sequentially on one GPU, not concurrently. Starts are aligned only in playback.

synchronized emission timestamps include prefill, decoding, and capture callback overhead; excludes loading and warmup. Recording synchronization and callbacks add overhead. EOS outputs remain shorter than the budget; token-limit outputs may end mid-answer.

Exact output checks against target-only: original: False; trained: False. BF16 divergences remain; no verified lossless-speedup claim is made.

GPU function wall time: 260.2s. Capture cache: `/cache/demo-recordings/demo-dcffd7998341`. This excludes separate CPU decoder checks and is not an invoice or a cost estimate.

No new training or W&B run was started. Model revisions, adapter SHA256, package versions, raw token IDs and per-emission events are in the JSON.

Reproduce playback: see [demo/README.md](../demo/README.md). A new capture requires the existing cached models and adapter; replay requires neither GPU nor credentials.
