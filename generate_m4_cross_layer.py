"""Compute layer-specific M4 verdict instability for Layers 1 and 2."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "m4"
PLOTS = ROOT / "plots" / "m4"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
K, N_PAIRS, CHUNK = 52, 250_000, 20_000
TOL = 1e-9
EPS = np.round(np.arange(0.25, 5.0001, 0.25), 10)
PHI = np.round(np.linspace(0.0, 1.0, 11), 10)
METRICS = ("Gini", "Theil", "Bad-specific Atkinson")


def get_weights(path):
    meta = pq.read_schema(path).metadata
    p = np.asarray(json.loads(meta[b"sampling"])["p"], dtype=np.float64)
    if p.shape != (K,) or abs(p.sum() - 1) > 1e-10:
        raise ValueError(f"Unexpected weights in {path}")
    return p


def metric_values(b, p, eps):
    mean = b @ p
    if np.any(mean <= 0):
        raise ValueError("Zero mean burden")
    r = b / mean[:, None]
    order = np.argsort(b, axis=1, kind="stable")
    sb = np.take_along_axis(b, order, axis=1)
    sp = np.take_along_axis(np.broadcast_to(p, b.shape), order, axis=1)
    g = np.sum(sp * sb * (2 * np.cumsum(sp, axis=1) - sp - 1), axis=1) / mean
    g /= 1 - p.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sum(p[None, :] * np.where(r > 0, r * np.log(np.maximum(r, 1e-300)), 0), axis=1)
    t /= np.log(1 / p.min())
    powers = 1 + eps
    raw = np.column_stack([((r ** power) @ p) ** (1 / power) - 1 for power in powers])
    maxima = p.min() ** (-eps / (1 + eps)) - 1
    return g, t, raw / maxima[None, :]


def verdict_signs(values):
    d = values[0::2] - values[1::2]
    return np.where(np.abs(d) <= TOL, 0, np.sign(d)).astype(np.int8)


def update_prob(prob, metric_idx, signs, weight):
    for cat, code in enumerate((-1, 0, 1)):
        prob[metric_idx, :, cat] += weight * (signs == code)


def m4_layer1(path, p):
    pf = pq.ParquetFile(path)
    cols = ["id"] + [f"b{k}" for k in range(1, K + 1)]
    total = np.zeros(3, dtype=np.float64)
    offset = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
        b = np.column_stack([batch[f"b{k}"].to_numpy(zero_copy_only=False) for k in range(1, K + 1)])
        n = len(b) // 2
        g, t, a = metric_values(b, p, EPS)
        prob = np.zeros((3, n, 3), dtype=np.float64)
        for j in range(len(EPS)):
            for m, values in enumerate((g, t, a[:, j])):
                update_prob(prob, m, verdict_signs(values), 1 / len(EPS))
        total += np.sum(1 - np.sum(prob * prob, axis=2), axis=1)
        offset += n
        print(f"Layer 1: {offset:,}/{N_PAIRS:,}", flush=True)
    if offset != N_PAIRS:
        raise ValueError(f"Expected {N_PAIRS} Layer 1 pairs; got {offset}")
    return np.maximum(total / N_PAIRS, 0.0)


def m4_layer2(path, p):
    pf = pq.ParquetFile(path)
    prefixes = ("cancelled", "mode_shifted", "anticipated", "postponed")
    cols = ["id"] + [f"{prefix}_{k}" for prefix in prefixes for k in range(1, K + 1)]
    scenarios = []
    scenarios += [("epsilon", float(e), 0.5, 0.5, 1 / (3 * len(EPS))) for e in EPS]
    scenarios += [("phi_mode", 1.0, float(pm), 0.5, 1 / (3 * len(PHI))) for pm in PHI]
    scenarios += [("phi_time", 1.0, 0.5, float(pt), 1 / (3 * len(PHI))) for pt in PHI]
    by_phi = {}
    for j, (_, _, pm, pt, _) in enumerate(scenarios):
        by_phi.setdefault((pm, pt), []).append(j)
    total = np.zeros(3, dtype=np.float64)
    offset = 0
    eps1 = int(np.flatnonzero(np.isclose(EPS, 1.0))[0])
    for batch in pf.iter_batches(batch_size=CHUNK, columns=cols):
        arrays = [np.column_stack([batch[f"{prefix}_{k}"].to_numpy(zero_copy_only=False)
                                   for k in range(1, K + 1)]).astype(np.float64, copy=False)
                  for prefix in prefixes]
        c, m, a, post = arrays
        n = len(c) // 2
        prob = np.zeros((3, n, 3), dtype=np.float64)
        for (pm, pt), ids in by_phi.items():
            b = c + pm * m + pt * (a + post)
            at_ref = np.isclose(pm, 0.5) and np.isclose(pt, 0.5)
            eps_eval = EPS if at_ref else np.array([1.0])
            g, t, atk = metric_values(b, p, eps_eval)
            for j in ids:
                axis, epsilon, _, _, weight = scenarios[j]
                ai = int(np.flatnonzero(np.isclose(eps_eval, epsilon if axis == "epsilon" else 1.0))[0])
                for mi, values in enumerate((g, t, atk[:, ai])):
                    update_prob(prob, mi, verdict_signs(values), weight)
        total += np.sum(1 - np.sum(prob * prob, axis=2), axis=1)
        offset += n
        print(f"Layer 2: {offset:,}/{N_PAIRS:,}", flush=True)
    if offset != N_PAIRS:
        raise ValueError(f"Expected {N_PAIRS} Layer 2 pairs; got {offset}")
    return np.maximum(total / N_PAIRS, 0.0)


def main():
    l1 = RESULTS / "seed42_K52_n250000_b.parquet"
    l2 = RESULTS / "seed42_K52_n250000_y.parquet"
    p = get_weights(l1)
    if not np.allclose(p, get_weights(l2), rtol=0, atol=1e-12):
        raise ValueError("Layer weights differ")
    u1 = m4_layer1(l1, p)
    u2 = m4_layer2(l2, p)
    with (RESULTS / "m3_m4" / "m3_m4_summary.csv").open(encoding="utf-8", newline="") as f:
        rows3 = {r["metric"]: r for r in csv.DictReader(f)}
    u3 = np.array([float(rows3[m]["normative_disagreement_pct"]) / 100 for m in METRICS])
    rows = [{"metric": m, "layer1_pct": 100*u1[i], "layer2_pct": 100*u2[i], "layer3_pct": 100*u3[i]}
            for i, m in enumerate(METRICS)]
    with (OUT / "m4_cross_layer.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tex = [
        r"\begin{table}[h!]", r"\centering", r"\small",
        r"\caption{M4 normative verdict instability by layer. Entries are the mean probabilities, in percent, that two independent draws from the layer-specific normative design give different verdict categories. Layer 1 varies $\varepsilon$ only; Layers 2 and 3 use the three-axis design. Layer 3 is averaged across 100 $c_{50}$ draws.}",
        r"\label{tab:m4_cross_layer}", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Metric & Layer 1 & Layer 2 & Layer 3 \\", r"\midrule",
    ]
    for row in rows:
        tex.append(f"{row['metric']} & {row['layer1_pct']:.2f} & {row['layer2_pct']:.2f} & {row['layer3_pct']:.2f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "m4_cross_layer.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    x = np.arange(3)
    width = 0.24
    series = (("Layer 1", [r["layer1_pct"] for r in rows], "#0072B2"),
              ("Layer 2", [r["layer2_pct"] for r in rows], "#D55E00"),
              ("Layer 3", [r["layer3_pct"] for r in rows], "#009E73"))
    for j, (label, vals, color) in enumerate(series):
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label, color=color)
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
    fig.savefig(PLOTS / "m4_cross_layer_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PLOTS / "m4_cross_layer_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(rows)


if __name__ == "__main__":
    main()
