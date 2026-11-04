"""Summarize and visualize the Layer 1 to Layer 2 shape-distribution shift.

Inputs are the cell-count exports produced by the M1 pipeline. The M2
distribution is based on all pairs, including Lorenz-dominance pairs. A single
Gini-vs-Theil row per cell is used because cell occupancy is comparison- and
epsilon-independent; duplicate rows from the other metric comparisons are not
counted again.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "m1"
OUT = ROOT / "results" / "m2"
PLOTS = ROOT / "plots" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

REF = (0.5, 0.5)
EDGES = np.linspace(0.0, 0.5, 11)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bin_key(row):
    return (round(float(row["z_low"]), 6), round(float(row["gamma_low"]), 6))


def load_layer1():
    rows = [r for r in read_rows(INPUT / "m1_cells_layer1.csv")
            if r["comparison"] == "Gini vs Theil"]
    counts = {bin_key(r): int(r["n_pairs"]) for r in rows}
    if len(counts) != 100 or sum(counts.values()) != 250_000:
        raise ValueError(f"Unexpected Layer 1 cell counts: {len(counts)} cells, {sum(counts.values())} pairs")
    return counts


def load_layer2():
    rows = [r for r in read_rows(INPUT / "m1_cells_layer2.csv")
            if r["comparison"] == "Gini vs Theil"]
    by_phi = {}
    for row in rows:
        phi = (round(float(row["phi_mode"]), 6), round(float(row["phi_time"]), 6))
        by_phi.setdefault(phi, {})[bin_key(row)] = int(row["n_pairs"])
    if len(by_phi) != 25:
        raise ValueError(f"Expected 25 Layer 2 weight settings, found {len(by_phi)}")
    for phi, counts in by_phi.items():
        if len(counts) != 100 or sum(counts.values()) != 250_000:
            raise ValueError(f"Unexpected Layer 2 cell counts at {phi}: {len(counts)} cells, {sum(counts.values())} pairs")
    return by_phi


def probability_grid(counts):
    grid = np.zeros((10, 10), dtype=float)
    for (z_low, gamma_low), count in counts.items():
        ix = min(9, int(round(z_low / 0.05)))
        iy = min(9, int(round(gamma_low / 0.05)))
        grid[iy, ix] = count / sum(counts.values())
    return grid


def main():
    l1_counts = load_layer1()
    l1 = probability_grid(l1_counts)
    layer2_counts = load_layer2()
    grid_by_phi = {phi: probability_grid(counts) for phi, counts in layer2_counts.items()}
    tv_by_phi = {phi: 0.5 * np.abs(grid - l1).sum() for phi, grid in grid_by_phi.items()}

    rows = []
    for (mode, time), tv in sorted(tv_by_phi.items()):
        rows.append({
            "phi_mode": f"{mode:.2f}",
            "phi_time": f"{time:.2f}",
            "total_variation": f"{tv:.8f}",
            "total_variation_pct": f"{100 * tv:.4f}",
            "is_reference": "true" if (mode, time) == REF else "false",
        })
    with (OUT / "m2_tv_by_phi.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Total variation distance between the Layer 1 and Layer 2 distributions over the shared $(Z,\Gamma)$ grid. Each entry is computed separately for one impact-weight setting; the reference setting is marked in bold.}",
        r"\label{tab:m2_tv}",
        r"\begin{tabular}{rrr}", r"\toprule",
        r"$\varphi_{\mathrm{mode}}$ & $\varphi_{\mathrm{time}}$ & TV (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        if row["is_reference"] == "true":
            line = (rf"\textbf{{{float(row['phi_mode']):.2f}}} & "
                    rf"\textbf{{{float(row['phi_time']):.2f}}} & "
                    rf"\textbf{{{float(row['total_variation_pct']):.2f}}} " + r"\\")
        else:
            line = (f"{float(row['phi_mode']):.2f} & {float(row['phi_time']):.2f} & "
                    f"{float(row['total_variation_pct']):.2f} " + r"\\")
        latex.append(line)
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "m2_tv_by_phi.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    ref_l2 = grid_by_phi[REF]
    delta_pp = 100 * (ref_l2 - l1)
    with (OUT / "m2_reference_cell_mass.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["z_low", "z_high", "gamma_low", "gamma_high", "layer1_mass_pct", "layer2_mass_pct", "delta_pp"])
        for iy in range(10):
            for ix in range(10):
                writer.writerow([f"{EDGES[ix]:.2f}", f"{EDGES[ix+1]:.2f}",
                                 f"{EDGES[iy]:.2f}", f"{EDGES[iy+1]:.2f}",
                                 f"{100*l1[iy,ix]:.6f}", f"{100*ref_l2[iy,ix]:.6f}",
                                 f"{delta_pp[iy,ix]:.6f}"])

    # Reference comparison: two mass maps and a signed difference map.
    vmax = max(float(l1.max()), float(ref_l2.max())) * 100
    dmax = max(abs(float(delta_pp.min())), abs(float(delta_pp.max())))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    for ax, data, title in [
        (axes[0], 100*l1, "Layer 1"),
        (axes[1], 100*ref_l2, r"Layer 2, $\varphi=(0.5,0.5)$"),
    ]:
        mesh = ax.pcolormesh(EDGES, EDGES, data, cmap="Blues", vmin=0, vmax=vmax,
                             shading="flat", edgecolors="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(r"Zheng descriptor $Z$")
        ax.set_ylabel(r"Lorenz intersection $\Gamma$")
        fig.colorbar(mesh, ax=ax, label="Share of all pairs in cell (%)")
    mesh = axes[2].pcolormesh(EDGES, EDGES, delta_pp, cmap="RdBu_r",
                              norm=TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax),
                              shading="flat", edgecolors="white", linewidth=0.5)
    axes[2].set_title(r"Change: Layer 2 $-$ Layer 1")
    axes[2].set_xlabel(r"Zheng descriptor $Z$")
    axes[2].set_ylabel(r"Lorenz intersection $\Gamma$")
    fig.colorbar(mesh, ax=axes[2], label="Change in mass (percentage points)")
    fig.suptitle("M2: pair-shape distribution at the reference impact weights", fontsize=13)
    fig.savefig(PLOTS / "m2_reference_shape_distribution.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "m2_reference_shape_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Sensitivity map: TV is calculated separately for each of the 25 settings.
    vals = np.zeros((5, 5), dtype=float)
    axis_values = [0, 0.25, 0.5, 0.75, 1]
    for iy, mode in enumerate(axis_values):
        for ix, time in enumerate(axis_values):
            vals[iy, ix] = 100 * tv_by_phi[(mode, time)]
    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    vmax = float(vals.max())
    mesh = ax.pcolormesh(np.arange(6), np.arange(6), vals, cmap="YlGnBu", vmin=0,
                         vmax=vmax, shading="flat", edgecolors="white", linewidth=1)
    for iy in range(5):
        for ix in range(5):
            ax.text(ix + 0.5, iy + 0.5, f"{vals[iy,ix]:.1f}", ha="center", va="center",
                    color="white" if vals[iy,ix] > 0.56*vmax else "black", fontsize=9)
    ax.add_patch(plt.Rectangle((2, 2), 1, 1, fill=False, edgecolor="#d62728", linewidth=2.5))
    ax.set_xticks(np.arange(5) + 0.5, [f"{x:.2f}" for x in axis_values])
    ax.set_yticks(np.arange(5) + 0.5, [f"{x:.2f}" for x in axis_values])
    ax.set_xlabel(r"$\varphi_{\mathrm{time}}$")
    ax.set_ylabel(r"$\varphi_{\mathrm{mode}}$")
    ax.set_title("Layer 1–Layer 2 total variation by impact weights")
    fig.colorbar(mesh, ax=ax, label="Total variation distance (%)")
    fig.savefig(PLOTS / "m2_tv_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "m2_tv_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Compact numeric diagnostics for the reference profile.
    i_max = np.unravel_index(np.argmax(delta_pp), delta_pp.shape)
    i_min = np.unravel_index(np.argmin(delta_pp), delta_pp.shape)
    print(f"Reference TV: {100*tv_by_phi[REF]:.4f}%")
    print(f"Layer 1 mass in Lorenz-dominance cell (first bin): {100*l1[0,0]:.4f}%")
    print(f"Layer 2 reference mass in Lorenz-dominance cell (first bin): {100*ref_l2[0,0]:.4f}%")
    print(f"Largest gain: {delta_pp[i_max]:.4f} pp in Z=[{EDGES[i_max[1]]:.2f},{EDGES[i_max[1]+1]:.2f}), Gamma=[{EDGES[i_max[0]]:.2f},{EDGES[i_max[0]+1]:.2f})")
    print(f"Largest loss: {delta_pp[i_min]:.4f} pp in Z=[{EDGES[i_min[1]]:.2f},{EDGES[i_min[1]+1]:.2f}), Gamma=[{EDGES[i_min[0]]:.2f},{EDGES[i_min[0]+1]:.2f})")
    print(f"TV range across settings: {100*min(tv_by_phi.values()):.4f}% to {100*max(tv_by_phi.values()):.4f}%")
    print(f"TV minimum settings: {[p for p, tv in tv_by_phi.items() if abs(tv-min(tv_by_phi.values())) < 1e-12]}")
    print(f"TV maximum settings: {[p for p, tv in tv_by_phi.items() if abs(tv-max(tv_by_phi.values())) < 1e-12]}")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    main()
