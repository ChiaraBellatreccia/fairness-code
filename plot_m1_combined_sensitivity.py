from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "m1" / "m1_overall_rates.csv"
OUT = ROOT / "plots" / "m1"
OUT.mkdir(parents=True, exist_ok=True)
REF_EPS = 1.0
REF_MODE = 0.5
REF_TIME = 0.5
ORDER = ["Gini vs Theil", "Gini vs Atkinson", "Theil vs Atkinson"]
COLORS = {"Gini vs Theil": "#0072B2", "Gini vs Atkinson": "#D55E00", "Theil vs Atkinson": "#009E73"}

rates = pd.read_csv(CSV)
rates["epsilon_num"] = pd.to_numeric(rates["epsilon"], errors="coerce")

fig = plt.figure(figsize=(6.8, 5.8))
gs = fig.add_gridspec(3, 6, height_ratios=[1.05, 1, 1])
axes = [
    fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
    fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6]),
    fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6]),
]


def draw(ax, layer, sweep, parameter=None):
    subset = rates[rates["layer"].eq(layer)].copy()
    if sweep == "epsilon":
        if layer != "Layer 1":
            subset = subset[subset["phi_mode"].round(6).eq(REF_MODE) & subset["phi_time"].round(6).eq(REF_TIME)]
        xs = np.sort(subset["epsilon_num"].dropna().unique())
        gt = subset[subset["comparison"].eq("Gini vs Theil")].iloc[0]
        ax.plot(xs, np.full(xs.shape, gt["disagree_pct"]), color=COLORS["Gini vs Theil"], lw=1.8, label="Gini--Theil")
        if layer == "Layer 3":
            sd = float(gt["epistemic_sd_pct"] or 0)
            if sd:
                ax.errorbar(xs, np.full(xs.shape, gt["disagree_pct"]), yerr=sd, fmt="none", color=COLORS["Gini vs Theil"], elinewidth=0.7, capsize=1.5)
        for comp in ORDER[1:]:
            series = subset[subset["comparison"].eq(comp)].sort_values("epsilon_num")
            y = series["disagree_pct"].to_numpy()
            sd = series["epistemic_sd_pct"].fillna(0).to_numpy()
            x = series["epsilon_num"].to_numpy()
            ax.plot(x, y, color=COLORS[comp], lw=1.8, marker="o", ms=2.8, label=comp.replace(" vs ", "--"))
            if layer == "Layer 3":
                ax.fill_between(x, y-sd, y+sd, color=COLORS[comp], alpha=0.14, linewidth=0)
                ax.errorbar(x, y, yerr=sd, fmt="none", color=COLORS[comp], elinewidth=0.65, capsize=1.4)
        ax.set_xlim(0.25, 5)
        ax.set_xticks([0.25, 1, 2.5, 5])
        ax.set_xlabel(r"Atkinson parameter $\varepsilon$", fontsize=9)
    else:
        other = "phi_time" if parameter == "phi_mode" else "phi_mode"
        subset = subset[subset["epsilon_num"].eq(REF_EPS) | subset["comparison"].eq("Gini vs Theil")]
        subset = subset[subset[other].round(6).eq(REF_MODE)]
        for comp in ORDER:
            series = subset[subset["comparison"].eq(comp)].sort_values("epsilon_num")
            x = series[parameter].to_numpy()
            y = series["disagree_pct"].to_numpy()
            sd = series["epistemic_sd_pct"].fillna(0).to_numpy()
            ax.plot(x, y, color=COLORS[comp], lw=1.8, marker="o", ms=3, label=comp.replace(" vs ", "--"))
            if layer == "Layer 3":
                ax.fill_between(x, y-sd, y+sd, color=COLORS[comp], alpha=0.14, linewidth=0)
                ax.errorbar(x, y, yerr=sd, fmt="none", color=COLORS[comp], elinewidth=0.65, capsize=1.4)
        xlabel = r"Impact weight $\varphi_{\mathrm{mode}}$" if parameter == "phi_mode" else r"Impact weight $\varphi_{\mathrm{time}}$"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlim(-0.035, 1.035)
    title = (r"$\varepsilon$ sweep" if sweep == "epsilon" else (r"$\varphi_{\mathrm{mode}}$ sweep" if sweep == "mode" else r"$\varphi_{\mathrm{time}}$ sweep"))
    ax.set_title(f"{title}: {layer}", fontsize=8.5, pad=2)
    ax.set_ylabel("Disagreement (%)" if ax in (axes[0], axes[3], axes[5]) else "", fontsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

for ax, layer in zip(axes[:3], ["Layer 1", "Layer 2", "Layer 3"]):
    draw(ax, layer, "epsilon")
draw(axes[3], "Layer 2", "mode", "phi_mode")
draw(axes[4], "Layer 3", "mode", "phi_mode")
draw(axes[5], "Layer 2", "time", "phi_time")
draw(axes[6], "Layer 3", "time", "phi_time")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.88, hspace=0.48, wspace=0.36)
fig.savefig(OUT / "m1_sensitivity_combined.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "m1_sensitivity_combined.png", dpi=240, bbox_inches="tight", pad_inches=0.03)