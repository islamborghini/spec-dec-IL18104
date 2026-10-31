"use strict";
(() => {
  const data = window.REPLAY_DATA;
  if (!data || data.illustrativePieces.length !== 2000) {
    throw new Error("Missing replay data. Run demo/build_data.py first.");
  }
  const $ = id => document.getElementById(id);
  const panels = [...document.querySelectorAll(".panel")].map(element => ({
    element, key: element.dataset.mode, output: element.querySelector(".output"),
    status: element.querySelector(".panel-status"), count: element.querySelector(".count"),
    fill: element.querySelector(".fill"), lastCount: -1,
  }));
  if (panels.length !== 4 || panels.some(p => !Number.isFinite(data.rates[p.key]) || data.rates[p.key] <= 0)) {
    throw new Error("Four panels with valid playback rates are required.");
  }
  let running = false, elapsed = 0, lastFrame = 0, frame = null;
  let illustration = [];
  for (const piece of data.illustrativePieces) {
    illustration.push((illustration.at(-1) || "") + piece);
  }
  illustration.unshift("");
  const isRecorded = () => $("mode").value === "recorded";
  const content = p => isRecorded() ? (data.recorded[p.key] || [""]) : illustration;

  function render() {
    let complete = true;
    for (const p of panels) {
      const text = content(p), total = text.length - 1;
      const count = Math.min(total, Math.floor(elapsed * data.rates[p.key]));
      complete = complete && count === total;
      if (p.lastCount !== count) {
        p.output.textContent = total === 0
          ? "Standalone drafter output was not recorded. Use illustrative mode to preview this lane."
          : count ? text[count] : "Waiting to start…";
        p.output.classList.toggle("waiting", count === 0);
        p.count.textContent = `${count.toLocaleString()} / ${total.toLocaleString()} tokens`;
        p.fill.style.width = `${total ? count / total * 100 : 0}%`;
        p.output.scrollTop = p.output.scrollHeight;
        p.lastCount = count;
      }
      p.status.textContent = total === 0 ? "Not recorded" : count === total ? "Replay complete"
        : running ? (isRecorded() ? "Replaying recorded output" : "Simulated stream")
        : elapsed ? "Paused" : "Ready";
    }
    const minutes = Math.floor(elapsed / 60);
    const seconds = (elapsed % 60).toFixed(1).padStart(4, "0");
    $("elapsed").textContent = `${String(minutes).padStart(2, "0")}:${seconds}`;
    if (complete && elapsed > 0) {
      running = false;
      $("status").textContent = "Replay complete";
      $("start").disabled = true;
      $("pause").disabled = true;
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
    running = false;
    cancelAnimationFrame(frame);
    elapsed = 0;
    panels.forEach(p => { p.lastCount = -1; });
    $("start").disabled = false;
    $("start").textContent = "Start replay";
    $("pause").disabled = true;
    $("status").textContent = "Ready";
    $("prompt").textContent = isRecorded() ? data.recordedPrompt : data.illustrativePrompt;
    $("badge").textContent = isRecorded() ? "RECORDED PLAYBACK · NOT LIVE" : "SIMULATED PLAYBACK · NOT LIVE";
    $("description").textContent = isRecorded()
      ? `Saved 64-token continuations · ${data.recordedArticle} · same prompt across recorded methods.`
      : "2,000-token illustration · the same authored answer in every panel · not recorded model output.";
    $("disclosure").textContent = isRecorded()
      ? "Real saved output, simulated streaming at aggregate benchmark rates. The standalone drafter has no recorded sample. No inference is running."
      : "Illustrative text and simulated streaming, not live inference. Reference rates were measured on 64-token validation runs; 2,000-token performance was not benchmarked. Base-drafter rate is illustrative.";
    render();
  }

  $("start").addEventListener("click", () => {
    if (running) return;
    running = true;
    lastFrame = performance.now();
    $("status").textContent = isRecorded() ? "Recorded playback" : "Illustrative playback";
    $("start").disabled = true;
    $("pause").disabled = false;
    render();
    frame = requestAnimationFrame(tick);
  });
  $("pause").addEventListener("click", () => {
    running = false;
    cancelAnimationFrame(frame);
    $("status").textContent = "Paused";
    $("start").textContent = "Resume replay";
    $("start").disabled = false;
    $("pause").disabled = true;
    render();
  });
  $("reset").addEventListener("click", reset);
  $("mode").addEventListener("change", reset);
  $("fullscreen").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      $("status").textContent = "Use the browser’s full-screen control";
    }
  });
  panels.forEach(p => {
    p.element.querySelector("[data-rate]").innerHTML = `${data.rates[p.key].toFixed(1)} <span>tok/s</span>`;
  });
  reset();
})();
