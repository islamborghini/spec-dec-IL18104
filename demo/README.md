# Four-panel comparison playback

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Open <http://127.0.0.1:8765>, click **Start replay**, and optionally use
**Full screen** while recording. **Pause**, **Resume**, and **Reset** control
playback. No GPU, model inference, API key, or external script is used.

The default mode streams the same **2,000-token authored illustrative text** in
four panels at constant playback rates. It is visibly labeled simulated and
is not evidence of 2,000-token model quality or performance. Target, original
speculative, and trained speculative reference rates come from the same saved
20-prompt A10 job, with k=4 and up to 64 output tokens. The standalone 0.6B
rate is **illustrative and unmeasured**, and is labeled that way.

Recorded mode replays the first validation prompt's actual target, original,
and trained outputs from `reports/training-to-10k-steps.json`. These are only
64 tokens. The base-drafter panel says no sample was recorded instead of
inventing a model output. Playback timing is simulated in both modes.

`data.js` is committed so playback works without Python dependencies. To rebuild
it from the existing reports and cached tokenizer:

```sh
.venv/bin/python demo/build_data.py
```

Keep the playback labels visible in a demo recording. Show the actual result
tables or W&B curves separately when explaining measured performance.
