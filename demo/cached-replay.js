(() => {
  'use strict';
  const data = window.CACHED_DEMO;
  const start = document.querySelector('#start');
  const pause = document.querySelector('#pause');
  const status = document.querySelector('#status');
  const panels = [...document.querySelectorAll('.panel')].map(el => {
    const row = data.models[el.dataset.mode];
    if (!row.events.length || !(row.seconds > 0)) throw new Error('Invalid recording');
    el.querySelector('[data-rate]').textContent = `${row.tokens_per_second.toFixed(1)} tok/s`;
    return {el, row, output: el.querySelector('.output'), cursor: 0, text: '', emitted: 0};
  });
  let running = false, elapsed = 0, lastFrame = 0, frame;
  function render() {
    document.querySelector('#elapsed').textContent = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${(elapsed % 60).toFixed(1).padStart(4, '0')}`;
    let complete = true;
    for (const p of panels) {
      while (p.cursor < p.row.events.length && p.row.events[p.cursor].at_s <= elapsed) {
        const event = p.row.events[p.cursor++];
        p.text = p.text.slice(0, event.keep) + event.append;
        p.emitted = event.token_count;
      }
      p.output.textContent = p.cursor ? p.text : 'Waiting for first recorded token…';
      p.output.classList.toggle('waiting', !p.cursor);
      p.output.scrollTop = p.output.scrollHeight;
      p.el.querySelector('.count').textContent = `${p.emitted.toLocaleString()} / ${p.row.tokens.toLocaleString()} tokens`;
      p.el.querySelector('.fill').style.width = `${100 * p.emitted / p.row.tokens}%`;
      const done = elapsed >= p.row.seconds;
      complete &&= done;
      p.el.querySelector('.panel-status').textContent = done
        ? `Complete · ${p.row.stop_reason === 'eos' ? 'EOS' : 'token limit'} · ${p.row.seconds.toFixed(1)}s`
        : running ? 'Replaying recorded timestamps' : elapsed ? 'Paused' : 'Ready';
    }
    if (complete) {
      running = false;
      status.textContent = 'Recording complete';
      start.disabled = true;
      pause.disabled = true;
    }
  }
  function tick(now) {
    if (!running) return;
    elapsed += Math.max(0, now - lastFrame) / 1000;
    lastFrame = now;
    render();
    if (running) frame = requestAnimationFrame(tick);
  }
  function reset() {
    cancelAnimationFrame(frame);
    running = false;
    elapsed = 0;
    for (const p of panels) Object.assign(p, {cursor: 0, text: '', emitted: 0});
    status.textContent = 'Ready · cached recording';
    start.textContent = 'Start replay';
    start.disabled = false;
    pause.disabled = true;
    render();
  }
  start.addEventListener('click', () => {
    running = true;
    lastFrame = performance.now();
    status.textContent = 'Recorded playback · 1×';
    start.disabled = true;
    pause.disabled = false;
    render();
    frame = requestAnimationFrame(tick);
  });
  pause.addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(frame);
    status.textContent = 'Paused';
    start.textContent = 'Resume replay';
    start.disabled = false;
    pause.disabled = true;
    render();
  });
  document.querySelector('#reset').addEventListener('click', reset);
  document.querySelector('#fullscreen').addEventListener('click', async () => {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  });
  document.querySelector('#prompt').textContent = data.prompt;
  document.querySelector('#description').textContent = `${data.gpu} · ${data.dtype} · ${data.prompt_tokens} prompt tokens · up to ${data.max_new_tokens.toLocaleString()} output tokens · average measured rates for this capture`;
  document.querySelector('#disclosure').textContent = 'Actual cached outputs and emission times, replayed at 1×. Methods ran separately on one GPU; their starts are aligned here. One demonstration prompt, not a repeated benchmark.';
  document.querySelector('#provenance').textContent = `Recording ${data.recording_id}, ${data.created_utc}, code ${data.code_revision}. Chat template with thinking disabled; k=${data.k}; selected adapter update ${data.selected_step}. Timings include prefill and recording overhead, exclude loading and warmup. Models may stop before 2,000 tokens. BF16 speculative outputs can diverge from target-only output; no verified lossless speedup is claimed. Raw tokens, timings and match checks: reports/demo-recording.json.`;
  reset();
})();
