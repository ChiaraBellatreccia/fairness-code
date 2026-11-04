"""Regenerate the M4 cross-layer chart from the computed CSV results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "m4" / "m4_cross_layer.csv"
OUT = ROOT / "plots" / "m4"
OUT.mkdir(parents=True, exist_ok=True)
LAYERS = ["Layer 1", "Layer 2", "Layer 3"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def main() -> None:
    df = pd.read_csv(SOURCE)
    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.24
    columns = ["layer1_pct", "layer2_pct", "layer3_pct"]
    for j, (label, col, color) in enumerate(zip(LAYERS, columns, COLORS)):
        bars = ax.bar(x + (j - 1) * width, df[col], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=10)
    ax.set_xticks(x, ["Gini", "Theil", "Bad-specific\nAtkinson"])
    ax.set_ylabel("Normative verdict instability (%)", fontsize=12.5)
    ax.set_title("M4 normative sensitivity across layers", fontsize=15)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_ylim(0, 20)
    ax.legend(frameon=False, ncols=3, fontsize=10.5, loc="upper center",
              bbox_to_anchor=(0.5, 0.98), borderaxespad=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "m4_cross_layer_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "m4_cross_layer_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
