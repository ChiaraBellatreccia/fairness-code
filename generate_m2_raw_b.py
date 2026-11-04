"""M2 raw-b distribution comparison for Layer 1 vs Layer 2.

Compares the population-weighted marginal distribution of raw group burden
values b in [0, 1], without computing pair-shape descriptors. A 10,000-bin
histogram approximates one-dimensional Wasserstein-1 distance; a resolution
check is written alongside the main outputs.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "m2_raw_b"
PLOTS = ROOT / "plots" / "m2_raw_b"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

L1_PATH = RESULTS / "seed42_K52_n250000_b.parquet"
L2_PATH = RESULTS / "seed42_K52_n250000_y.parquet"
GROUPS_PATH = RESULTS / "seed42_K52_groups.csv"
N_BINS = 10_000
WEIGHTS = [0, 0.25, 0.5, 0.75, 1]
REF = (0.5, 0.5)
CHUNK = 25_000


def read_weights():
    with GROUPS_PATH.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    p = np.array([float(r["p"]) for r in rows])
    if p.size != 52 or not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError(f"Unexpected population weights: K={p.size}, sum={p.sum()}")
    return p


def matrix_from_batch(batch, names):
    return np.column_stack([
        batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
        for name in names
    ])


def hist_by_group(matrix, out_hist):
    for k in range(matrix.shape[1]):
        idx = np.minimum((matrix[:, k] * N_BINS).astype(np.int64), N_BINS - 1)
        out_hist[k] += np.bincount(idx, minlength=N_BINS)


def load_layer1_hist():
    parquet = pq.ParquetFile(L1_PATH)
    b_names = [f"b{k}" for k in range(1, 53)]
    hist = np.zeros((52, N_BINS), dtype=np.int64)
    sums = np.zeros(52, dtype=np.float64)
    n = 0
    for batch in parquet.iter_batches(columns=b_names, batch_size=CHUNK):
        b = matrix_from_batch(batch, b_names)
        hist_by_group(b, hist)
        sums += b.sum(axis=0)
        n += b.shape[0]
    return hist, sums, n


def load_layer2_hists():
    parquet = pq.ParquetFile(L2_PATH)
    prefixes = ["cancelled", "mode_shifted", "anticipated", "postponed"]
    names = {prefix: [f"{prefix}_{k}" for k in range(1, 53)] for prefix in prefixes}
    configs = [(mode, time) for mode in WEIGHTS for time in WEIGHTS]
    hist = np.zeros((len(configs), 52, N_BINS), dtype=np.int64)
    sums = np.zeros((len(configs), 52), dtype=np.float64)
    n = 0
    columns = sum((names[prefix] for prefix in prefixes), [])
    for batch in parquet.iter_batches(columns=columns, batch_size=CHUNK):
        c, m, a, post = [matrix_from_batch(batch, names[prefix]) for prefix in prefixes]
        time_shift = a + post
        for i, (mode_weight, time_weight) in enumerate(configs):
            b = c + mode_weight * m + time_weight * time_shift
            hist_by_group(b, hist[i])
            sums[i] += b.sum(axis=0)
        n += c.shape[0]
    return configs, hist, sums, n


def weighted_mass(hist, weights, n_rows):
    mass = weights @ (hist.astype(np.float64) / n_rows)
    return mass / mass.sum()


def w1_from_hist(mass_a, mass_b, bin_width):
    cdf_a = np.cumsum(mass_a)
    cdf_b = np.cumsum(mass_b)
    return float(np.abs(cdf_a[:-1] - cdf_b[:-1]).sum() * bin_width)


def weighted_mean(sums, weights, n_rows):
    return float(weights @ (sums / n_rows))


def main():
    p = read_weights()
    h1, sums1, n1 = load_layer1_hist()
    configs, h2, sums2, n2 = load_layer2_hists()
    if n1 != n2 or n1 != 500_000:
        raise ValueError(f"Expected 500,000 burden vectors per layer; got L1={n1}, L2={n2}")

    mass1 = weighted_mass(h1, p, n1)
    masses2 = [weighted_mass(h2[i], p, n2) for i in range(len(configs))]
    w1 = np.array([w1_from_hist(mass1, m, 1 / N_BINS) for m in masses2])
    means1 = weighted_mean(sums1, p, n1)
    means2 = np.array([weighted_mean(sums2[i], p, n2) for i in range(len(configs))])

    rows = []
    for (mode, time), dist, mean2 in zip(configs, w1, means2):
        rows.append({
            "phi_mode": f"{mode:.2f}",
            "phi_time": f"{time:.2f}",
            "weighted_wasserstein1_raw_b": f"{dist:.8f}",
            "weighted_wasserstein1_raw_b_pct_of_unit_range": f"{100*dist:.5f}",
            "layer1_weighted_mean_b": f"{means1:.8f}",
            "layer2_weighted_mean_b": f"{mean2:.8f}",
            "mean_b_change_layer2_minus_layer1": f"{mean2-means1:.8f}",
            "is_reference": "true" if (mode, time) == REF else "false",
        })
    with (OUT / "m2_raw_b_by_phi.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Population-weighted Wasserstein-1 distance between the marginal distributions of raw burden values $b$ in Layers 1 and 2. The reference impact weights are marked in bold. Distances are in units of $b\in[0,1]$.}",
        r"\label{tab:m2_raw_b}", r"\begin{tabular}{rrr}", r"\toprule",
        r"$\varphi_{\mathrm{mode}}$ & $\varphi_{\mathrm{time}}$ & $W_1(b)$ \\", r"\midrule",
    ]
    for row in rows:
        mode, time = float(row["phi_mode"]), float(row["phi_time"])
        dist = float(row["weighted_wasserstein1_raw_b"])
        if row["is_reference"] == "true":
            line = rf"\textbf{{{mode:.2f}}} & \textbf{{{time:.2f}}} & \textbf{{{dist:.4f}}} " + r"\\"
        else:
            line = f"{mode:.2f} & {time:.2f} & {dist:.4f} " + r"\\"
        latex.append(line)
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "m2_raw_b_by_phi.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    # Reference CDF and signed CDF difference. This uses raw b, not pair shapes.
    iref = configs.index(REF)
    mass_ref = masses2[iref]
    edges = np.linspace(0, 1, N_BINS + 1)
    x = edges[1:]
    cdf1 = np.cumsum(mass1)
    cdf2 = np.cumsum(mass_ref)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(x, cdf1, color="#0072B2", lw=2, label="Layer 1")
    axes[0].plot(x, cdf2, color="#D55E00", lw=2, label=r"Layer 2, $\varphi=(0.5,0.5)$")
    axes[0].set_xlabel("Raw burden value $b$")
    axes[0].set_ylabel("Population-weighted cumulative share")
    axes[0].set_title("Burden distributions")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False)
    axes[1].plot(x, cdf2 - cdf1, color="#6A3D9A", lw=2)
    axes[1].axhline(0, color="#555555", lw=0.8)
    axes[1].set_xlabel("Raw burden value $b$")
    axes[1].set_ylabel(r"$F_2(b)-F_1(b)$")
    axes[1].set_title(f"CDF difference; $W_1={w1[iref]:.4f}$")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.suptitle("M2: Layer 1 to Layer 2 shift in raw burden values", fontsize=13)
    fig.savefig(PLOTS / "m2_raw_b_reference_cdf.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "m2_raw_b_reference_cdf.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Sensitivity map of the raw-b W1 distance over all 25 weight settings.
    values = w1.reshape(5, 5)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    vmax = float(values.max())
    mesh = ax.pcolormesh(np.arange(6), np.arange(6), values, cmap="YlGnBu",
                         vmin=0, vmax=vmax, shading="flat", edgecolors="white", linewidth=1)
    for iy in range(5):
        for ix in range(5):
            ax.text(ix+0.5, iy+0.5, f"{values[iy,ix]:.3f}", ha="center", va="center",
                    color="white" if values[iy,ix] > 0.56*vmax else "black", fontsize=9)
    ax.add_patch(plt.Rectangle((2, 2), 1, 1, fill=False, edgecolor="#d62728", linewidth=2.5))
    ax.set_xticks(np.arange(5)+0.5, [f"{v:.2f}" for v in WEIGHTS])
    ax.set_yticks(np.arange(5)+0.5, [f"{v:.2f}" for v in WEIGHTS])
    ax.set_xlabel(r"$\varphi_{\mathrm{time}}$")
    ax.set_ylabel(r"$\varphi_{\mathrm{mode}}$")
    ax.set_title(r"Raw-b $W_1$ distance by impact weights")
    fig.colorbar(mesh, ax=ax, label=r"$W_1$ in raw burden units")
    fig.savefig(PLOTS / "m2_raw_b_w1_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "m2_raw_b_w1_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Resolution sensitivity for the chosen 10,000-bin approximation.
    resolutions = [1000, 5000, 10000]
    res_rows = []
    for nb in resolutions:
        def coarsen(mass):
            return mass.reshape(nb, N_BINS // nb).sum(axis=1)
        d = w1_from_hist(coarsen(mass1), coarsen(mass_ref), 1/nb)
        res_rows.append((nb, d))
    with (OUT / "m2_raw_b_resolution_check.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["histogram_bins", "reference_w1_raw_b"])
        writer.writerows([[n, f"{d:.8f}"] for n, d in res_rows])

    print(f"Burden vectors per layer: {n1:,}")
    print(f"Population-weighted mean raw b, Layer 1: {means1:.6f}")
    print(f"Population-weighted mean raw b, Layer 2 reference: {means2[iref]:.6f}")
    print(f"Mean raw-b change at reference: {means2[iref]-means1:+.6f}")
    print(f"Reference raw-b W1: {w1[iref]:.6f}")
    print(f"Raw-b W1 range over settings: {w1.min():.6f} to {w1.max():.6f}")
    print("Raw-b W1 resolution check:", ", ".join(f"{n} bins={d:.6f}" for n, d in res_rows))
    print("Minimum settings:", [configs[i] for i in np.flatnonzero(np.isclose(w1, w1.min()))])
    print("Maximum settings:", [configs[i] for i in np.flatnonzero(np.isclose(w1, w1.max()))])


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    main()
