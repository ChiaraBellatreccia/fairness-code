"""Regenerate the M3/M4/M5 summary figure from computed summary rates."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "m3_m4" / "m3_m4_summary.csv"
OUT = ROOT / "plots" / "m3_m4"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ["Gini", "Theil", "Bad-specific Atkinson"]


def main() -> None:
    df = pd.read_csv(SOURCE).set_index("metric").loc[METRICS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.subplots_adjust(top=0.73, bottom=0.18, left=0.10, right=0.98, wspace=0.34)
    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, df["epistemic_disagreement_pct"], width,
                color="#0072B2", label="Epistemic (M3)")
    axes[0].bar(x + width / 2, df["normative_disagreement_pct"], width,
                color="#D55E00", label="Normative (M4)")
    metric_labels = ["Gini", "Theil", "Bad-specific\nAtkinson"]
    axes[0].set_xticks(x, metric_labels)
    axes[0].set_ylabel("Two-draw verdict disagreement (%)", fontsize=12.5)
    axes[0].set_title("Comparable instability rates", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=10.5)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylim(0, 12)

    e_share = df["epistemic_shapley_pct_of_joint"].to_numpy()
    n_share = df["normative_shapley_pct_of_joint"].to_numpy()
    axes[1].bar(x, e_share, color="#0072B2", label="Epistemic")
    axes[1].bar(x, n_share, bottom=e_share, color="#D55E00", label="Normative")
    axes[1].set_xticks(x, metric_labels)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Share of joint verdict variance (%)", fontsize=12.5)
    axes[1].set_title("Shapley attribution (shares sum to 100%)", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=10.5)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.legend(handles=[Patch(facecolor="#0072B2", label="Epistemic"),
                        Patch(facecolor="#D55E00", label="Normative")],
               frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncols=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "m3_m4_layer3_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "m3_m4_layer3_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
