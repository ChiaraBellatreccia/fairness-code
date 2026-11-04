from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "m1"
PLOTS = ROOT / "plots" / "m1"
PLOTS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Gini vs Theil": "#0072B2",
    "Gini vs Atkinson": "#D55E00",
    "Theil vs Atkinson": "#009E73",
}
ORDER = ["Gini vs Theil", "Gini vs Atkinson", "Theil vs Atkinson"]
LAYERS = ["Layer 1", "Layer 2", "Layer 3"]
REF_MODE = 0.5
REF_TIME = 0.5
REF_EPS = 1.0

rates = pd.read_csv(RESULTS / "m1_overall_rates.csv")
rates["epsilon_num"] = pd.to_numeric(rates["epsilon"], errors="coerce")
contexts = pd.read_csv(RESULTS / "m1_context_rates.csv")
cells = pd.read_csv(RESULTS / "m1_cells_layer1.csv")
cells = pd.concat(
    [cells, pd.read_csv(RESULTS / "m1_cells_layer2.csv"), pd.read_csv(RESULTS / "m1_cells_layer3.csv")],
    ignore_index=True,
)
cells["epsilon_num"] = pd.to_numeric(cells["epsilon"], errors="coerce")


def reference_rows(layer: str) -> pd.DataFrame:
    subset = rates[rates["layer"].eq(layer)]
    if layer != "Layer 1":
        subset = subset[
            subset["phi_mode"].round(6).eq(REF_MODE)
            & subset["phi_time"].round(6).eq(REF_TIME)
        ]
    return subset[
        ((subset["comparison"].eq("Gini vs Theil")) & subset["epsilon_num"].isna())
        | (subset["comparison"].isin(["Gini vs Atkinson", "Theil vs Atkinson"])
           & subset["epsilon_num"].eq(REF_EPS))
    ]


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PLOTS / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PLOTS / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_reference_heatmap() -> None:
    selected = cells[
        ((cells["comparison"].eq("Gini vs Theil")) & cells["epsilon_num"].isna())
        | (cells["comparison"].isin(["Gini vs Atkinson", "Theil vs Atkinson"])
           & cells["epsilon_num"].eq(REF_EPS))
    ].copy()
    selected = selected[
        selected["layer"].eq("Layer 1")
        | (selected["phi_mode"].round(6).eq(REF_MODE) & selected["phi_time"].round(6).eq(REF_TIME))
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 10.8), sharex=True, sharey=True)
    norm = Normalize(vmin=0, vmax=40)
    cmap = plt.get_cmap("Blues")
    edges = np.linspace(0, 0.5, 11)

    for r, layer in enumerate(LAYERS):
        for c, comparison in enumerate(ORDER):
            ax = axes[r, c]
            rows = selected[selected["layer"].eq(layer) & selected["comparison"].eq(comparison)]
            z = np.full((10, 10), np.nan)
            sparse = []
            for row in rows.itertuples(index=False):
                iz = min(9, int(round(row.z_low / 0.05)))
                ig = min(9, int(round(row.gamma_low / 0.05)))
                non_dominance_n = int(row.n_pairs) - int(row.n_dominance)
                if non_dominance_n > 0:
                    z[ig, iz] = float(row.disagree_pct_excl_dominance)
                    if non_dominance_n < 50:
                        sparse.append((row.z_low + 0.025, row.gamma_low + 0.025))
            mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(z), cmap=cmap, norm=norm,
                                 shading="flat", edgecolors="white", linewidth=0.6)
            if sparse:
                sx, sy = zip(*sparse)
                ax.scatter(sx, sy, s=11, color="#333333", zorder=4)
            summary = reference_rows(layer)
            pct = summary.loc[summary["comparison"].eq(comparison), "disagree_pct"].iloc[0]
            ax.set_title(f"{comparison}\n{pct:.2f}% of pairs", fontsize=12.5, pad=7)
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 0.5)
            ax.set_xticks([0, 0.25, 0.5])
            ax.set_yticks([0, 0.25, 0.5])
            ax.tick_params(labelsize=9)
            if r == 2:
                ax.set_xlabel(r"Zheng descriptor $Z$", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{layer}\n$\Gamma$", fontsize=11)
    fig.suptitle(r"M1 disagreement at $\varepsilon=1$; Layers 2--3 use $\varphi=(0.5,0.5)$",
                 fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.91, bottom=0.08, wspace=0.12, hspace=0.25)
    cax = fig.add_axes([0.93, 0.17, 0.018, 0.67])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Disagreement rate within cell (%)", fontsize=10.5)
    fig.text(0.5, 0.02,
             "Lorenz-dominance pairs are excluded from cells; dots mark cells with fewer than 50 non-dominance pairs. "
             "Layer 3 pools 100 c50 realizations and uses whole-day trips as the denominator.",
             ha="center", fontsize=9.5)
    save_figure(fig, "m1_reference_heatmap_eps1_phi0.5_0.5")


def plot_epsilon_sweep() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), sharey=True)
    eps = np.sort(rates["epsilon_num"].dropna().unique())
    for ax, layer in zip(axes, LAYERS):
        subset = rates[rates["layer"].eq(layer)]
        if layer != "Layer 1":
            subset = subset[subset["phi_mode"].round(6).eq(REF_MODE)
                            & subset["phi_time"].round(6).eq(REF_TIME)]
        gt_row = subset[subset["comparison"].eq("Gini vs Theil")].iloc[0]
        gt = gt_row["disagree_pct"]
        ax.plot(eps, np.full_like(eps, gt, dtype=float), color=COLORS["Gini vs Theil"],
                lw=2, label="Gini vs Theil")
        if layer == "Layer 3":
            gt_sd = float(gt_row["epistemic_sd_pct"])
            ax.errorbar(eps, np.full_like(eps, gt, dtype=float),
                        yerr=np.full_like(eps, gt_sd, dtype=float), fmt="none",
                        ecolor=COLORS["Gini vs Theil"], elinewidth=0.7, capsize=1.7, alpha=0.8)
        for comparison in ["Gini vs Atkinson", "Theil vs Atkinson"]:
            series = subset[subset["comparison"].eq(comparison)].sort_values("epsilon_num")
            y = series["disagree_pct"].to_numpy()
            sd = series["epistemic_sd_pct"].fillna(0).to_numpy()
            ax.plot(series["epsilon_num"], y, color=COLORS[comparison], lw=2, marker="o",
                    ms=3.2, label=comparison)
            if layer == "Layer 3":
                ax.fill_between(series["epsilon_num"].to_numpy(), y - sd, y + sd,
                                color=COLORS[comparison], alpha=0.16, linewidth=0)
                ax.errorbar(series["epsilon_num"], y, yerr=sd, fmt="none", ecolor=COLORS[comparison],
                            elinewidth=0.7, capsize=1.7, alpha=0.8)
        ax.set_title(layer + ("\n" + r"$\varphi=(0.5,0.5)$" if layer != "Layer 1" else ""), fontsize=12.5)
        ax.set_xlabel(r"Atkinson parameter $\varepsilon$", fontsize=12)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_xlim(0.25, 5)
        ax.set_xticks([0.25, 1, 2, 3, 4, 5])
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Pairwise disagreement rate (%)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.86))
    fig.text(0.5, -0.01,
             "Layer 3 shaded bands/error bars show ±1 standard deviation across 100 c50 realizations; "
             "they are not sampling confidence intervals.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    save_figure(fig, "m1_pairwise_vs_epsilon")


def plot_phi_sweep(parameter: str) -> None:
    other = "phi_time" if parameter == "phi_mode" else "phi_mode"
    other_label = r"$\varphi_{\mathrm{time}}$" if other == "phi_time" else r"$\varphi_{\mathrm{mode}}$"
    x_label = r"$\varphi_{\mathrm{mode}}$" if parameter == "phi_mode" else r"$\varphi_{\mathrm{time}}$"
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True)
    x_values = np.array([0, 0.25, 0.5, 0.75, 1.0])

    for ax, layer in zip(axes, ["Layer 2", "Layer 3"]):
        subset = rates[rates["layer"].eq(layer)]
        subset = subset[subset["epsilon_num"].eq(REF_EPS) | subset["comparison"].eq("Gini vs Theil")]
        subset = subset[subset[other].round(6).eq(REF_MODE)]
        for comparison in ORDER:
            series = subset[subset["comparison"].eq(comparison)].copy()
            series = series.sort_values(parameter)
            x = series[parameter].to_numpy()
            y = series["disagree_pct"].to_numpy()
            sd = series["epistemic_sd_pct"].fillna(0).to_numpy()
            ax.plot(x, y, color=COLORS[comparison], lw=2, marker="o", ms=4, label=comparison)
            if layer == "Layer 3":
                ax.fill_between(x, y - sd, y + sd, color=COLORS[comparison], alpha=0.16, linewidth=0)
                ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=COLORS[comparison],
                            elinewidth=0.8, capsize=2)
        ax.set_title(f"{layer}: epsilon=1, other weight=0.5", fontsize=12)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_xticks(x_values)
        ax.set_xlim(-0.04, 1.04)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Pairwise disagreement rate (%)", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.86))
    title_param = r"$\varphi_{\mathrm{mode}}$" if parameter == "phi_mode" else r"$\varphi_{\mathrm{time}}$"
    fig.text(0.5, -0.02,
             "Layer 3 shaded bands/error bars show ±1 standard deviation across 100 c50 realizations; "
             "they are not sampling confidence intervals.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    save_figure(fig, f"m1_pairwise_vs_{parameter}")


def write_reference_table() -> None:
    row_order = ORDER
    layer_values = {}
    for layer in LAYERS:
        rows = reference_rows(layer)
        layer_values[layer] = {row.comparison: row for row in rows.itertuples(index=False)}

    table_rows = []
    for comparison in row_order:
        l1 = layer_values["Layer 1"][comparison]
        l2 = layer_values["Layer 2"][comparison]
        l3 = layer_values["Layer 3"][comparison]
        label = comparison
        if comparison != "Gini vs Theil":
            label += r", $\varepsilon=1$"
        table_rows.append({
            "comparison": label,
            "Layer 1 (%)": f"{l1.disagree_pct:.2f}",
            "Layer 2 (%)": f"{l2.disagree_pct:.2f}",
            "Layer 3 mean (SD) (%)": f"{l3.disagree_pct:.2f} ({l3.epistemic_sd_pct:.2f})",
        })
    frame = pd.DataFrame(table_rows)
    frame.to_csv(RESULTS / "m1_reference_pairwise_eps1_phi0.5_0.5.csv", index=False)
    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Pairwise disagreement rates at the reference settings, as percentages of all sampled pairs. Atkinson comparisons use $\varepsilon=1$; Layers 2 and 3 use $\varphi_{\mathrm{mode}}=\varphi_{\mathrm{time}}=0.5$. Layer 3 entries report the pooled rate and the standard deviation across 100 $c_{50}$ realizations.}",
        r"\label{tab:m1_reference_pairwise}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric comparison & Layer 1 & Layer 2 & Layer 3 mean (SD) " + r"\\",
        r"\midrule",
    ]
    for row in table_rows:
        latex.append(f"{row['comparison']} & {row['Layer 1 (%)']} & {row['Layer 2 (%)']} & {row['Layer 3 mean (SD) (%)']} " + r"\\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (RESULTS / "m1_reference_pairwise_eps1_phi0.5_0.5.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    write_reference_table()
    plot_reference_heatmap()
    plot_epsilon_sweep()
    plot_phi_sweep("phi_mode")
    plot_phi_sweep("phi_time")
