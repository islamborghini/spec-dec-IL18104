# Four-panel recorded inference

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Open <http://127.0.0.1:8765>, click **Start replay**, and optionally use
**Full screen** while recording. Pause, Resume, and Reset control playback.
No GPU or credentials are needed to replay the committed assets.

All four panels use actual model outputs and synchronized emission timestamps
from one A10 capture. Starts are aligned for comparison; inference was performed
**sequentially**, not concurrently. Playback runs at 1×, preserves speculative
bursts, and shows each method's measured average tokens/sec. The shared prompt
has a 2,000-token generation budget; EOS can stop a model earlier. Outputs are
never padded or replaced with authored text.

The standalone base is Qwen3-0.6B; target-only is Qwen3-4B; both speculative
methods use k=4, before training versus the selected update-9,875 adapter.
This chat-formatted prompt is a demonstration, not the WikiText validation
benchmark. BF16 output mismatches remain possible: inspect the recorded match
checks before making any claim about equivalence. Recording adds synchronization
and callback overhead; timings include prefill but exclude loading and warmup.

Raw token IDs, events, model pins, adapter hash and runtime versions are saved in
`reports/demo-recording.json`. See `reports/DEMO_RECORDING.md` for the results.
To rebuild browser assets from that recording and the locally cached tokenizer:

```sh
.venv/bin/python demo/build_data.py
```

To make a **new paid GPU recording** (requires authorized Modal credentials,
cached models and the selected adapter in the existing volume, and clean git):

```sh
MODAL_PROFILE=islam-95550 .venv/bin/python record_demo.py
.venv/bin/python demo/build_data.py
```

The capture performs decoder checks first, runs one bounded A10 job, and saves
the raw output locally and in the Modal volume. It creates no W&B training run.
Keep the recorded-playback label visible in the video. Show the validation
results and training curves separately when discussing the broader experiment.
