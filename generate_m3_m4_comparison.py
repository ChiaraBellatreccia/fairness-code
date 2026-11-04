"""Matched Layer 3 comparison of epistemic and normative verdict instability.

The normative design is a reference-anchored, one-factor-at-a-time design:
epsilon in {0.25, ..., 5.0} at (phi_mode, phi_time)=(0.5,0.5), and each phi
parameter on {0,0.1,...,1} with the other phi fixed at 0.5. For the aggregate
comparison, first select one of these three parameter axes with equal
probability, then select a value uniformly along that axis. This is a declared
sensitivity-design weighting, not a stakeholder probability distribution.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from metrics import beta_grid


ROOT = Path(__file__).resolve().parent
L3 = ROOT / "results" / "layer3"
DRAWS = L3 / "seed42_n25000_y"
OUT = ROOT / "results" / "m3_m4"
PLOTS = ROOT / "plots" / "m3_m4"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

PREFIXES = ("cancelled", "mode_shifted", "anticipated", "postponed")
K = 52
N_DRAWS = 100
TIE_TOL = 1e-9
PHI_GRID = np.round(np.linspace(0.0, 1.0, 11), 10)
EPSILONS = np.sort(-beta_grid())
METRICS = ("Gini", "Theil", "Bad-specific Atkinson")
METRIC_LABELS = ("Gini", "Theil", "Bad-specific Atkinson")


def c50_from_name(path: Path) -> float:
    match = re.search(r"c50-([0-9.]+)_y\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse c50 from {path.name}")
    return float(match.group(1))


def load_outcomes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = ["id"] + [f"{prefix}_{j}" for prefix in PREFIXES for j in range(1, K + 1)]
    table = pq.read_table(path, columns=cols)
    ids = table["id"].to_numpy(zero_copy_only=False)
    if len(ids) != 2 * 25_000 or not np.array_equal(ids, np.repeat(np.arange(25_000), 2)):
        raise ValueError(f"Unexpected pair layout in {path.name}")
    arrays = []
    for prefix in PREFIXES:
        arrays.append(np.column_stack([
            table[f"{prefix}_{j}"].to_numpy(zero_copy_only=False) for j in range(1, K + 1)
        ]).astype(np.float64, copy=False))
    return tuple(arrays)  # type: ignore[return-value]


def metric_values(b: np.ndarray, p: np.ndarray, epsilons: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row-wise Gini, Theil, and bad-Atkinson values at each epsilon."""
    mean = b @ p
    if np.any(mean <= 0):
        raise ValueError("Encountered a zero mean burden")
    r = b / mean[:, None]
    order = np.argsort(b, axis=1, kind="stable")
    sorted_b = np.take_along_axis(b, order, axis=1)
    sorted_p = np.take_along_axis(np.broadcast_to(p, b.shape), order, axis=1)
    gini = np.sum(sorted_p * sorted_b * (2 * np.cumsum(sorted_p, axis=1) - sorted_p - 1), axis=1) / mean
    gini /= 1 - p.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        theil = np.sum(p[None, :] * np.where(r > 0, r * np.log(np.maximum(r, 1e-300)), 0.0), axis=1)
    theil /= np.log(1 / p.min())
    # Each column is one epsilon; beta=-epsilon and power=1+epsilon.
    powers = 1.0 + epsilons
    raw = np.column_stack([((r ** power) @ p) ** (1.0 / power) - 1.0 for power in powers])
    maxima = p.min() ** (-epsilons / (1.0 + epsilons)) - 1.0
    atkinson = raw / maxima[None, :]
    return gini, theil, atkinson


def signs_of_differences(values: np.ndarray) -> np.ndarray:
    """Convert row-wise paired scores to pair verdicts in {-1,0,+1}."""
    d = values[0::2] - values[1::2]
    return np.where(np.abs(d) <= TIE_TOL, 0, np.sign(d)).astype(np.int8)


def one_hot_counts(signs: np.ndarray) -> np.ndarray:
    """Counts of {-1,0,+1} per pair for an array of signs."""
    return np.column_stack([(signs == s).astype(np.uint16) for s in (-1, 0, 1)])


def gini_impurity(prob: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(prob * prob, axis=-1)


def main() -> None:
    paths = sorted(DRAWS.glob("seed42_K52_n25000_c50-*_y.parquet"), key=c50_from_name)
    if len(paths) != N_DRAWS:
        raise ValueError(f"Expected {N_DRAWS} epistemic files; found {len(paths)}")
    meta = json.loads(pq.read_schema(paths[0]).metadata[b"sampling"])
    p = np.asarray(meta["p"], dtype=np.float64)
    if len(p) != K or abs(p.sum() - 1) > 1e-10:
        raise ValueError("Unexpected population weights")

    # The 42 one-at-a-time normative scenarios are weighted as a mixture of
    # three axes, with uniform weights within each axis (not uniform over rows).
    scenarios: list[tuple[str, float, float, float, float]] = []
    for eps in EPSILONS:
        scenarios.append(("epsilon", float(eps), 0.5, 0.5, 1.0 / (3 * len(EPSILONS))))
    for phi in PHI_GRID:
        scenarios.append(("phi_mode", 1.0, float(phi), 0.5, 1.0 / (3 * len(PHI_GRID))))
    for phi in PHI_GRID:
        scenarios.append(("phi_time", 1.0, 0.5, float(phi), 1.0 / (3 * len(PHI_GRID))))
    n_scen = len(scenarios)
    scenario_weights = np.array([x[4] for x in scenarios], dtype=np.float64)
    if not np.isclose(scenario_weights.sum(), 1.0):
        raise RuntimeError("Normative scenario weights do not sum to one")
    scenarios_by_phi: dict[tuple[float, float], list[int]] = {}
    for j, (_, _, pm, pt, _) in enumerate(scenarios):
        scenarios_by_phi.setdefault((pm, pt), []).append(j)

    # Counts of verdict categories conditional on normative scenario, across E.
    counts_given_n = np.zeros((3, n_scen, 25_000, 3), dtype=np.uint16)
    # Conditional categorical probabilities given each epistemic draw, averaged
    # over the declared normative design.
    probs_given_e = np.zeros((3, N_DRAWS, 25_000, 3), dtype=np.float32)
    c50s = []

    for e_idx, path in enumerate(paths):
        c50s.append(c50_from_name(path))
        cancelled, mode, anticipated, postponed = load_outcomes(path)
        for (phi_mode, phi_time), scenario_ids in scenarios_by_phi.items():
            b = cancelled + phi_mode * mode + phi_time * (anticipated + postponed)
            at_reference_phi = abs(phi_mode - 0.5) < 1e-9 and abs(phi_time - 0.5) < 1e-9
            eps_eval = EPSILONS if at_reference_phi else np.array([1.0])
            gini, theil, atkinson = metric_values(b, p, eps_eval)
            v_g = signs_of_differences(gini[:, None])[:, 0]
            v_t = signs_of_differences(theil[:, None])[:, 0]
            v_atk_by_column = [signs_of_differences(atkinson[:, [col]])[:, 0]
                               for col in range(atkinson.shape[1])]
            for j in scenario_ids:
                axis, eps, _, _, weight = scenarios[j]
                if axis == "epsilon":
                    atk_col = int(np.flatnonzero(np.isclose(EPSILONS, eps))[0])
                else:
                    atk_col = int(np.flatnonzero(np.isclose(eps_eval, 1.0))[0])
                for m, signs in enumerate((v_g, v_t, v_atk_by_column[atk_col])):
                    cats = one_hot_counts(signs)
                    counts_given_n[m, j] += cats
                    probs_given_e[m, e_idx] += (weight * cats).astype(np.float32)
        del cancelled, mode, anticipated, postponed
        print(f"Processed epistemic draw {e_idx + 1}/{N_DRAWS}: c50={c50s[-1]:.4f}", flush=True)

    rows = []
    details = {}
    for m, metric in enumerate(METRIC_LABELS):
        # Epistemic disagreement, averaging over normative scenarios.
        p_e_given_n = counts_given_n[m].astype(np.float64) / N_DRAWS
        ue_by_n = gini_impurity(p_e_given_n).mean(axis=1)
        ue = float(np.dot(scenario_weights, ue_by_n))
        # Normative disagreement, averaging over epistemic draws.
        p_n_given_e = probs_given_e[m].astype(np.float64)
        un = float(gini_impurity(p_n_given_e).mean())
        p_joint = p_n_given_e.mean(axis=0)
        total = float(gini_impurity(p_joint).mean())
        # Shapley allocation of categorical verdict variance (Gini impurity).
        # For two independent input groups E and N, c(E)=T-U_N and c(N)=T-U_E.
        shapley_e = 0.5 * (total - un + ue)
        shapley_n = 0.5 * (total - ue + un)
        rows.append({
            "metric": metric,
            "n_policy_pairs": 25_000,
            "n_epistemic_draws": N_DRAWS,
            "n_normative_axis_scenarios": n_scen,
            "epistemic_disagreement_pct": 100 * ue,
            "normative_disagreement_pct": 100 * un,
            "joint_verdict_uncertainty_pct": 100 * total,
            "epistemic_shapley_pct_of_joint": 100 * shapley_e / total if total > 0 else np.nan,
            "normative_shapley_pct_of_joint": 100 * shapley_n / total if total > 0 else np.nan,
            "normative_design": "equal mixture of epsilon, phi_mode, phi_time one-factor sweeps; uniform within each axis",
            "tie_treatment": "tie is a third verdict category",
        })
        details[metric] = {"ue": ue, "un": un, "total": total, "shapley_e": shapley_e, "shapley_n": shapley_n,
                           "ue_by_n": ue_by_n, "scenario_weights": scenario_weights}

    with (OUT / "m3_m4_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "m3_m4_design.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "value", "phi_mode", "phi_time", "weight"])
        for axis, eps, pm, pt, weight in scenarios:
            writer.writerow([axis, eps if axis == "epsilon" else "", pm, pt, weight])

    # M3 fragility at each normative setting, for the one-at-a-time plots.
    with (OUT / "m3_normative_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "normative_axis", "value", "epistemic_fragility_pct", "design_weight"])
        for metric in METRIC_LABELS:
            for j, (axis, eps, pm, pt, weight) in enumerate(scenarios):
                value = eps if axis == "epsilon" else (pm if axis == "phi_mode" else pt)
                writer.writerow([metric, axis, value, 100 * details[metric]["ue_by_n"][j], weight])

    # M3 counterpart to the M1 one-parameter-at-a-time plots: epistemic
    # fragility is evaluated at each normative value, averaging across c50.
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True, constrained_layout=True)
    axes_specs = [
        ("epsilon", r"Atkinson parameter $\varepsilon$", EPSILONS),
        ("phi_mode", r"Impact weight $\varphi_{\mathrm{mode}}$", PHI_GRID),
        ("phi_time", r"Impact weight $\varphi_{\mathrm{time}}$", PHI_GRID),
    ]
    colors = ("#0072B2", "#D55E00", "#009E73")
    for ax, (axis_name, x_label, grid) in zip(axes, axes_specs):
        ids = [j for j, scenario in enumerate(scenarios) if scenario[0] == axis_name]
        xvals = [scenarios[j][1] if axis_name == "epsilon" else
                 (scenarios[j][2] if axis_name == "phi_mode" else scenarios[j][3]) for j in ids]
        for color, metric in zip(colors, METRIC_LABELS):
            yvals = 100 * details[metric]["ue_by_n"][ids]
            ax.plot(xvals, yvals, marker="o", markersize=3.2, linewidth=1.5,
                    color=color, label=metric)
        ax.set_xlabel(x_label)
        ax.set_xticks(grid)
        if len(grid) > 12:
            ax.set_xticks(grid[::4])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Epistemic fragility (%)")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(PLOTS / "m3_normative_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(PLOTS / "m3_normative_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # LaTeX table for the harmonized comparison.
    tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Layer 3 verdict instability under epistemic and normative variation. Epistemic and normative rates use the same 25,000 policy pairs and treat ties as a third verdict category. Normative configurations follow the equally weighted mixture of the one-factor-at-a-time sweeps described in the text. Shapley shares allocate joint categorical verdict variance between the two sources.}",
        r"\label{tab:m3_m4_layer3}", r"\begin{tabular}{lrrrrr}", r"\toprule",
        "Metric & Epistemic (\\%) & Normative (\\%) & Joint (\\%) & Epistemic share (\\%) & Normative share (\\%) " + r"\\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(f"{r['metric']} & {r['epistemic_disagreement_pct']:.2f} & {r['normative_disagreement_pct']:.2f} & {r['joint_verdict_uncertainty_pct']:.2f} & {r['epistemic_shapley_pct_of_joint']:.1f} & {r['normative_shapley_pct_of_joint']:.1f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "m3_m4_summary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Compact comparison plot: rates are directly comparable; Shapley shares
    # provide an additive attribution that accounts for E x N interaction.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.subplots_adjust(top=0.73, bottom=0.18, left=0.10, right=0.98, wspace=0.34)
    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, [100 * details[m]["ue"] for m in METRIC_LABELS], width,
                color="#0072B2", label="Epistemic (M3)")
    axes[0].bar(x + width / 2, [100 * details[m]["un"] for m in METRIC_LABELS], width,
                color="#D55E00", label="Normative (M4)")
    axes[0].set_xticks(x, ["Gini", "Theil", "Bad-specific\nAtkinson"])
    axes[0].set_ylabel("Two-draw verdict disagreement (%)", fontsize=12.5)
    axes[0].set_title("Comparable instability rates", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=10.5)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylim(0, 12)
    e_share = [details[m]["shapley_e"] / details[m]["total"] * 100 for m in METRIC_LABELS]
    n_share = [details[m]["shapley_n"] / details[m]["total"] * 100 for m in METRIC_LABELS]
    axes[1].bar(x, e_share, color="#0072B2", label="Epistemic")
    axes[1].bar(x, n_share, bottom=e_share, color="#D55E00", label="Normative")
    axes[1].set_xticks(x, ["Gini", "Theil", "Bad-specific\nAtkinson"])
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
    fig.savefig(PLOTS / "m3_m4_layer3_comparison.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PLOTS / "m3_m4_layer3_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Textual methods/results subsections ready to adapt in the paper.
    latex_sections = r"""\subsection{Epistemic uncertainty and normative sensitivity (M3--M4)}
\label{sec:m3_m4}

We examine whether pairwise policy verdicts are unstable under epistemic uncertainty (M3) or normative choices (M4). The analysis uses the same 25,000 matched Layer~3 policy pairs and the same three inequality metrics throughout. For a metric $M$, let $V_{i,M}(e,n)\in\{-1,0,+1\}$ denote its verdict for policy pair $i$ under epistemic realization $e$ and normative configuration $n$; zero denotes a tie (absolute verdict difference at most $10^{-9}$). A positive value indicates that the first policy has the larger inequality value, and a negative value indicates the reverse.

For either source of variation, we summarize instability by the probability that two independent draws of that source give different verdict categories for the same pair. If $p_{i,M}(s\mid n)$ is the empirical probability of category $s$ over epistemic draws at normative configuration $n$, and $q_{i,M}(s\mid e)$ is its probability over normative configurations at epistemic realization $e$, then
\[
U_E(M)=\frac{1}{N}\sum_{i=1}^{N}\sum_n w_n\left[1-\sum_s p_{i,M}(s\mid n)^2\right],\qquad
U_N(M)=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|\mathcal E|}\sum_{e\in\mathcal E}\left[1-\sum_s q_{i,M}(s\mid e)^2\right].
\]
M3 reports $U_E$ and M4 reports $U_N$. Both rates therefore have the same interpretation and scale; they describe sensitivity under the specified sampling designs, not probabilities of real-world stakeholder choices.

\paragraph{M3: Epistemic uncertainty.}
For each normative configuration, M3 compares the metric's verdict across independent draws from the sampled epistemic distribution for $c_{50}$. We then average the pair-level disagreement over the normative design. This extends the reference-configuration analysis at $\varepsilon=1$ and $\varphi=(0.5,0.5)$ by evaluating epistemic instability across normative choices, so that M3 can be compared with M4 on a common basis.

\paragraph{M4: Normative sensitivity.}
M4 applies the same pair-level disagreement calculation to normative configurations, holding each epistemic realization fixed and averaging over the 100 $c_{50}$ draws. The normative design is a reference-anchored one-factor-at-a-time sweep: $\varepsilon\in\{0.25,0.50,\ldots,5.00\}$ with $\varphi_{\mathrm{mode}}=\varphi_{\mathrm{time}}=0.5$; $\varphi_{\mathrm{mode}}\in\{0,0.1,\ldots,1\}$ with $\varphi_{\mathrm{time}}=0.5$ and $\varepsilon=1$; and $\varphi_{\mathrm{time}}\in\{0,0.1,\ldots,1\}$ with $\varphi_{\mathrm{mode}}=0.5$ and $\varepsilon=1$. The summary uses an equal mixture over the three parameter sweeps and uniform weights within each sweep. This is a design-based weighting chosen to summarize the sensitivity analysis; it is not an elicited distribution over stakeholder preferences.

The two disagreement rates need not sum to total verdict uncertainty because epistemic and normative variation can interact. Let $T_M$ be the mean, over policy pairs, of $1-\sum_s p_{i,M}(s)^2$ under the joint epistemic--normative design. We additionally report a two-factor Shapley allocation of this joint categorical verdict variance:
\[
S_E(M)=\tfrac12\big[(T_M-U_N(M))+U_E(M)\big],\qquad
S_N(M)=\tfrac12\big[(T_M-U_E(M))+U_N(M)\big].
\]
Verdict categories are represented by three indicator variables, whose covariance trace equals the categorical Gini impurity $1-\sum_s p(s)^2$. The two contributions sum to $T_M$; the reported shares are $S_E/T_M$ and $S_N/T_M$. For all three metrics in this sample, M4 is larger than M3 (9.36--10.03\% versus 3.02--3.20\%), and the Shapley allocation assigns 79.1--80.6\% of joint verdict variance to normative variation. These are conclusions under the stated normative design weighting. The rates and Shapley shares are reported in Table~\ref{tab:m3_m4_layer3} and visualized in Figure~\ref{fig:m3_m4_layer3}.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{plots/m3_m4/m3_m4_layer3_comparison.pdf}
    \caption{Layer~3 comparison of epistemic verdict instability (M3) and normative sensitivity (M4), computed on matched policy pairs. Left: probability that two independent draws of the varied source give different verdict categories. Right: Shapley allocation of joint categorical verdict variance; shares sum to 100\%.}
    \label{fig:m3_m4_layer3}
\end{figure}
"""
    (OUT / "m3_m4_subsections.tex").write_text(latex_sections, encoding="utf-8")
    print("SUMMARY")
    for row in rows:
        print(row["metric"], f"M3={row['epistemic_disagreement_pct']:.4f}%",
              f"M4={row['normative_disagreement_pct']:.4f}%",
              f"joint={row['joint_verdict_uncertainty_pct']:.4f}%",
              f"Shapley E/N={row['epistemic_shapley_pct_of_joint']:.2f}/{row['normative_shapley_pct_of_joint']:.2f}%")
    print(f"Wrote outputs under {OUT} and {PLOTS}")


if __name__ == "__main__":
    main()
