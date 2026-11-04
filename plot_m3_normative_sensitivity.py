"""Plot the M3 fragility rates computed for each normative configuration."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "m3_m4" / "m3_normative_sensitivity.csv"
OUT = ROOT / "plots" / "m3_m4"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {
    "Gini": "#0072B2",
    "Theil": "#D55E00",
    "Bad-specific Atkinson": "#009E73",
}
SPECS = {
    "epsilon": (r"Atkinson parameter $\varepsilon$", r"$\varepsilon$",
                "m3_epistemic_fragility_vs_epsilon"),
    "phi_mode": (r"Impact weight $\varphi_{\mathrm{mode}}$", r"$\varphi_{\mathrm{mode}}$",
                 "m3_epistemic_fragility_vs_phi_mode"),
    "phi_time": (r"Impact weight $\varphi_{\mathrm{time}}$", r"$\varphi_{\mathrm{time}}$",
                 "m3_epistemic_fragility_vs_phi_time"),
}


def main():
    df = pd.read_csv(SOURCE)
    for axis, (label, short, filename) in SPECS.items():
        part = df[df["normative_axis"] == axis]
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        for metric, color in COLORS.items():
            sub = part[part["metric"] == metric].sort_values("value")
            ax.plot(sub["value"], sub["epistemic_fragility_pct"], marker="o",
                    markersize=3.8, linewidth=1.7, color=color, label=metric)
        ax.set_xlabel(label, fontsize=12.5)
        ax.set_ylabel("Epistemic fragility (%)", fontsize=12.5)
        ax.set_title(f"M3 fragility as {short} varies", fontsize=15)
        if axis == "epsilon":
            ax.set_xticks(part["value"].drop_duplicates().sort_values()[::4])
        else:
            ax.set_xticks(part["value"].drop_duplicates().sort_values())
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.legend(frameon=False, fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(OUT / f"{filename}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # Combined three-panel version for the paper. Each panel uses its own
    # y-axis scale so small changes are visible within each parameter sweep.
    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 4.2), sharex=False, sharey=False,
        constrained_layout=True,
    )
    panel_specs = [
        ("epsilon", r"Atkinson parameter $\varepsilon$", r"$\varepsilon$"),
        ("phi_mode", r"Impact weight $\varphi_{\mathrm{mode}}$",
         r"$\varphi_{\mathrm{mode}}$"),
        ("phi_time", r"Impact weight $\varphi_{\mathrm{time}}$",
         r"$\varphi_{\mathrm{time}}$"),
    ]
    for ax, (axis, xlabel, short) in zip(axes, panel_specs):
        part = df[df["normative_axis"] == axis]
        for metric, color in COLORS.items():
            sub = part[part["metric"] == metric].sort_values("value")
            ax.plot(sub["value"], sub["epistemic_fragility_pct"], marker="o",
                    markersize=3.8, linewidth=1.7, color=color, label=metric)
        ax.set_xlabel(xlabel, fontsize=11.5)
        ax.set_ylabel("Epistemic fragility (%)", fontsize=10.5)
        ax.set_title(f"As {short} varies", fontsize=12.5)
        values = part["value"].drop_duplicates().sort_values()
        ax.set_xticks(values[::4] if axis == "epsilon" else values)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=9, loc="best")
    fig.savefig(OUT / "m3_normative_sensitivity.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "m3_normative_sensitivity.png", dpi=220,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
