"""Render the pilot's recorded losses without rerunning training."""

import json
import math
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    report = json.loads((root / "reports/training.json").read_text())
    histories = [report["train_history"], report["validation_history"]]
    assert len(histories[0]) == 125 and len(histories[1]) == 6
    assert all(math.isfinite(row["kl"]) for history in histories for row in history)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for ax, history, title in zip(axes, histories, ["Training batches", "Held-out validation chunks"]):
        ax.plot([row["step"] for row in history], [row["kl"] for row in history],
                linewidth=1.5, marker="o" if len(history) < 10 else None, markersize=4)
        ax.set(title=title, xlabel="Optimizer update", ylabel="KL(target || drafter) / position")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    best = next(row for row in histories[1] if row["step"] == report["best_step"])
    axes[1].scatter(best["step"], best["kl"], marker="*", s=120, color="#e58c20",
                    zorder=3, label=f"Selected: step {best['step']}")
    axes[1].legend(fontsize=8)
    figure.suptitle("WikiText pilot: one epoch, rank-8 LoRA", fontsize=12)
    output = root / "reports/training_loss.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
