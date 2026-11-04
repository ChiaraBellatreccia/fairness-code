"""Compute M3 epistemic fragility for the matched Layer 3 policy-pair sample.

Reference: c50=7.5, evaluated on the same 25,000 policy pairs as the 100
uncertainty realizations. Normative profile: phi_mode=phi_time=0.5 and
bad-specific Atkinson epsilon=1 (beta=-1 in metrics.py).
"""
from pathlib import Path
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.special import xlogy

from metrics import gini_norm, theil_norm, bad_atkinson_norm


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
L3 = RESULTS / "layer3"
REFERENCE = L3 / "seed42_K52_n25000_y.parquet"
DRAWS_DIR = L3 / "seed42_n25000_y"
OUT = RESULTS / "m3"
PLOTS = ROOT / "plots" / "m3"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

PHI_MODE = 0.5
PHI_TIME = 0.5
EPSILON = 1.0
BETA = -EPSILON  # project bad-specific Atkinson parameterization
TIE_TOL = 1e-9
BATCH_ROWS = 10_000
METRICS = ["Gini", "Theil", "Bad-specific Atkinson (epsilon=1)"]
OUTCOME_PREFIXES = ["cancelled", "mode_shifted", "anticipated", "postponed"]


def metadata(path):
    schema = pq.read_schema(path)
    return json.loads(schema.metadata[b"sampling"])


def c50_from_name(path):
    match = re.search(r"c50-([0-9.]+)_y\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse c50 from {path.name}")
    return float(match.group(1))


def burden_from_batch(batch):
    k = 52
    cols = {}
    for prefix in OUTCOME_PREFIXES:
        names = [f"{prefix}_{j}" for j in range(1, k + 1)]
        cols[prefix] = np.column_stack([
            batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
            for name in names
        ])
    return cols["cancelled"] + PHI_MODE * cols["mode_shifted"] + PHI_TIME * (cols["anticipated"] + cols["postponed"])


def normalized_metric_values(b, p):
    """Rows of b -> normalized Gini, Theil, and bad-Atkinson epsilon=1."""
    means = b @ p
    if np.any(means <= 0):
        raise ValueError("M3 found a zero mean burden; metric is undefined")
    relative = b / means[:, None]

    order = np.argsort(b, axis=1, kind="stable")
    sorted_b = np.take_along_axis(b, order, axis=1)
    sorted_p = np.broadcast_to(p, b.shape)
    sorted_p = np.take_along_axis(sorted_p, order, axis=1)
    cumulative_p = np.cumsum(sorted_p, axis=1)
    gini = np.sum(sorted_p * sorted_b * (2 * cumulative_p - sorted_p - 1), axis=1) / means
    gini /= (1 - p.min())

    theil = np.sum(p * xlogy(relative, relative), axis=1)
    theil /= np.log(1 / p.min())

    # For epsilon=1 in the bad-specific index, beta=-1 and power order is 2.
    atkinson = np.sqrt((relative ** 2) @ p) - 1
    atkinson_max = p.min() ** (BETA / (1 - BETA)) - 1
    atkinson /= atkinson_max
    return np.column_stack([gini, theil, atkinson])


def verdicts(path, p, expected_ids=None):
    parquet = pq.ParquetFile(path)
    ids = []
    out = []
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
        b = burden_from_batch(batch)
        if b.shape[0] % 2:
            raise ValueError(f"Odd row count in {path.name}")
        ids.append(batch.column(batch.schema.get_field_index("id")).to_numpy())
        values = normalized_metric_values(b, p)
        # The project format stores A and B as consecutive rows for each pair.
        out.append(values[0::2] - values[1::2])
    ids = np.concatenate(ids)
    v = np.vstack(out)
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError(f"Policy-pair IDs/order differ in {path.name}")
    if not np.array_equal(ids, np.repeat(np.arange(len(ids) // 2), 2)):
        raise ValueError(f"Unexpected pair-id layout in {path.name}")
    return ids, v


def validate_vectorized_metrics(p, ref_path):
    table = pq.read_table(ref_path, columns=["id"] + [f"{o}_{k}" for o in OUTCOME_PREFIXES for k in range(1, 53)],
                          use_threads=True)
    row = table.slice(0, 2)
    b = burden_from_batch(row)
    vector = normalized_metric_values(b, p)
    direct = np.array([
        [gini_norm(p, b[i]), theil_norm(p, b[i]), bad_atkinson_norm(p, b[i], BETA)]
        for i in range(2)
    ])
    if np.max(np.abs(vector - direct)) > 1e-10:
        raise RuntimeError(f"Vectorized metric calculation mismatch: {np.max(np.abs(vector-direct)):.3g}")


def main():
    if not REFERENCE.is_file():
        raise FileNotFoundError(f"Missing matched c50=7.5 reference output: {REFERENCE}")
    draw_paths = sorted(DRAWS_DIR.glob("seed42_K52_n25000_c50-*_y.parquet"), key=c50_from_name)
    if len(draw_paths) != 100:
        raise ValueError(f"Expected 100 c50 draws, found {len(draw_paths)}")

    ref_meta = metadata(REFERENCE)
    if ref_meta.get("K") != 52 or ref_meta.get("n_samples") != 25_000:
        raise ValueError(f"Unexpected reference metadata: K={ref_meta.get('K')}, n={ref_meta.get('n_samples')}")
    p = np.asarray(ref_meta["p"], dtype=float)
    if abs(p.sum() - 1) > 1e-10:
        raise ValueError(f"Population weights sum to {p.sum()}")
    reference_c50 = ref_meta.get("c50", ref_meta.get("twin", {}).get("c50"))
    if reference_c50 != 7.5:
        raise ValueError(f"Reference file is not c50=7.5: {reference_c50}")
    if any(metadata(path).get("K") != 52 for path in draw_paths):
        raise ValueError("A draw does not use K=52")

    validate_vectorized_metrics(p, REFERENCE)
    ref_ids, v_ref = verdicts(REFERENCE, p)
    ref_sign = np.where(np.abs(v_ref) <= TIE_TOL, 0, np.sign(v_ref)).astype(np.int8)
    reference_ties = np.sum(ref_sign == 0, axis=0)
    eligible = ref_sign != 0
    n_eligible = eligible.sum(axis=0)

    flips = np.zeros((len(draw_paths), 3), dtype=np.int64)
    ties = np.zeros((len(draw_paths), 3), dtype=np.int64)
    per_pair_flips = np.zeros((len(draw_paths), v_ref.shape[0], 3), dtype=np.uint8)
    per_pair_ties = np.zeros_like(per_pair_flips)
    c50_values = []
    for j, path in enumerate(draw_paths):
        c50 = c50_from_name(path)
        c50_values.append(c50)
        _, v = verdicts(path, p, expected_ids=ref_ids)
        sign = np.where(np.abs(v) <= TIE_TOL, 0, np.sign(v)).astype(np.int8)
        for m in range(3):
            opposite = eligible[:, m] & (sign[:, m] == -ref_sign[:, m])
            tie_here = eligible[:, m] & (sign[:, m] == 0)
            flips[j, m] = opposite.sum()
            ties[j, m] = tie_here.sum()
            per_pair_flips[j, :, m] = opposite
            per_pair_ties[j, :, m] = tie_here
        print(f"Processed c50={c50:.4f} ({j+1}/100)", flush=True)

    n_draws = len(draw_paths)
    total_eligible = n_eligible * n_draws
    total_flips = flips.sum(axis=0)
    total_ties = ties.sum(axis=0)
    summary_rows = []
    for m, metric in enumerate(METRICS):
        pair_rates = per_pair_flips[:, :, m].sum(axis=0) / n_draws
        pair_tie_rates = per_pair_ties[:, :, m].sum(axis=0) / n_draws
        summary_rows.append({
            "metric": metric,
            "n_policy_pairs": int(v_ref.shape[0]),
            "reference_tied_pairs_excluded": int(reference_ties[m]),
            "reference_nontied_pairs": int(n_eligible[m]),
            "n_reference_reversals": int(total_flips[m]),
            "reversal_rate_all_draws_pct": f"{100*total_flips[m]/total_eligible[m]:.4f}",
            "n_epistemic_ties": int(total_ties[m]),
            "epistemic_tie_rate_pct_of_eligible_draws": f"{100*total_ties[m]/total_eligible[m]:.4f}",
            "reversal_rate_non_tied_draws_pct": f"{100*total_flips[m]/(total_eligible[m]-total_ties[m]):.4f}" if total_eligible[m] > total_ties[m] else "NA",
            "mean_pair_reversal_fraction_pct": f"{100*pair_rates.mean():.4f}",
            "median_pair_reversal_fraction_pct": f"{100*np.median(pair_rates):.4f}",
            "p90_pair_reversal_fraction_pct": f"{100*np.quantile(pair_rates, 0.90):.4f}",
            "pairs_reversed_in_at_least_one_draw_pct": f"{100*np.mean(pair_rates > 0):.4f}",
            "reference_c50": "7.5",
            "normative_profile": "epsilon=1, phi_mode=0.5, phi_time=0.5",
        })
    with (OUT / "m3_layer3_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Long-form pair-level rates for further analyses and M5.
    with (OUT / "m3_layer3_pair_rates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "metric", "reference_verdict_sign", "reference_tied",
                         "n_c50_reversals", "reversal_fraction_all_100", "n_c50_ties", "tie_fraction_all_100"])
        for pair_id in range(v_ref.shape[0]):
            for m, metric in enumerate(METRICS):
                writer.writerow([pair_id, metric, int(ref_sign[pair_id, m]), ref_sign[pair_id, m] == 0,
                                 int(per_pair_flips[:, pair_id, m].sum()),
                                 f"{per_pair_flips[:, pair_id, m].mean():.4f}",
                                 int(per_pair_ties[:, pair_id, m].sum()),
                                 f"{per_pair_ties[:, pair_id, m].mean():.4f}"])

    # Per-realization rates, useful for checking whether the aggregate is driven by a few c50 values.
    with (OUT / "m3_layer3_by_c50.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["c50", *[f"{metric}_reversal_pct" for metric in METRICS], *[f"{metric}_tie_pct" for metric in METRICS]])
        for j, c50 in enumerate(c50_values):
            rev = [f"{100*flips[j,m]/n_eligible[m]:.5f}" for m in range(3)]
            tie = [f"{100*ties[j,m]/n_eligible[m]:.5f}" for m in range(3)]
            writer.writerow([f"{c50:.4f}", *rev, *tie])

    # LaTeX table with pooled reversal and tie rates.
    latex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{M3 at Layer 3. For each policy pair, verdicts at the $c_{50}=7.5$ reference are compared directly with 100 epistemic realizations. Reversal and tie rates use reference-nontied pairs and all 100 draws as the denominator.}",
        r"\label{tab:m3_layer3}", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Metric & Eligible pairs & Reversal rate (\%) & Epistemic ties (\%) \\", r"\midrule",
    ]
    for row in summary_rows:
        latex.append(f"{row['metric']} & {row['reference_nontied_pairs']} & {row['reversal_rate_all_draws_pct']} & {row['epistemic_tie_rate_pct_of_eligible_draws']} " + r"\\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "m3_layer3_summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    # Clear summary: rates at each epistemic draw plus fraction of pairs ever reversed.
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.8, 1]})
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "^"]
    for m, metric in enumerate(METRICS):
        rates = 100 * flips[:, m] / n_eligible[m]
        axes[0].plot(c50_values, rates, color=colors[m], marker=markers[m], ms=2.8,
                     lw=1.1, label=metric, alpha=0.9)
    axes[0].axvline(7.5, color="#555555", ls="--", lw=1.1, label=r"Reference $c_{50}=7.5$")
    axes[0].set_xlabel(r"Epistemic realization $c_{50}$")
    axes[0].set_ylabel("Reference-nontied pairs reversed (%)")
    axes[0].set_title("Reversal rate at each draw")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    any_reversal = [float(row["pairs_reversed_in_at_least_one_draw_pct"]) for row in summary_rows]
    short_labels = ["Gini", "Theil", "Bad Atkinson\n(eps=1)"]
    axes[1].bar(short_labels, any_reversal, color=colors, width=0.68)
    for i, value in enumerate(any_reversal):
        axes[1].text(i, value + 0.18, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, max(any_reversal) + 1.3)
    axes[1].set_ylabel("Pairs with at least one reversal (%)")
    axes[1].set_title("Across all 100 draws")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.suptitle(r"M3 Layer 3: verdict reversals from the $c_{50}=7.5$ reference", fontsize=13)
    fig.savefig(PLOTS / "m3_layer3_reversal_summary.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "m3_layer3_reversal_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Reference policy pairs: {v_ref.shape[0]:,}; c50 draws: {n_draws}; c50 range: {min(c50_values):.4f}–{max(c50_values):.4f}")
    for row in summary_rows:
        print(f"{row['metric']}: reversal={row['reversal_rate_all_draws_pct']}%, ties={row['epistemic_tie_rate_pct_of_eligible_draws']}%, "
              f"pairs with >=1 reversal={row['pairs_reversed_in_at_least_one_draw_pct']}%")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    main()
