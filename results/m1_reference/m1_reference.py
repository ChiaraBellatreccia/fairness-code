"""M1 reference table and figures, recomputed from the samples with the project's own functions.

Nothing is read from the summary CSVs in results/m1/: they are only compared with the results at the end (m1_reference_checks.txt).

Inputs (paths relative to the layer1 folder; hashes in input_manifest.csv)
    Layer 1  results/seed42_K52_n250000_b.parquet        250,000 pairs of burden vectors, columns b1 ... b52
    Layer 2  results/seed42_K52_n250000_y.parquet        250,000 pairs of outcome shares, columns <outcome>_<k>
    Layer 3  results/layer3/seed42_n25000_y/*_y.parquet  25,000 policy pairs x 100 c50 realizations, one file per c50

Reference profile (illustrative, not universal and not empirically elicited): Atkinson epsilon = 1; phi_mode = phi_time = 0.5 in
Layers 2 and 3. Layer 1 has no phi parameters. In the code, w_mode = phi_mode and w_time = phi_time.

What is computed
    at the reference profile      the three layers, all 41 comparisons (Gini vs Theil and the 20 epsilon values) and the (Z, Gamma) cells
    on the phi grid, Layers 2, 3  the cross product of PHI_GRID for phi_mode and phi_time, at epsilon = 1 only (Gini vs Theil, Gini vs
                                  Atkinson, Theil vs Atkinson). Change PHI_STEP to change the grid.

Project functions used
    impact_function.linear_impact_function   b_k = c_k + phi_mode m_k + phi_time (a_k + p_k)
    metrics._prepare_rows, _gini_rows, _bad_atkinson_rows, _gini_max, _theil_max, _bad_atkinson_max, beta_grid
                                             the same steps as metrics.compute_metrics, which would rewrite the input file.
                                             Atkinson is bad_atkinson with beta = -epsilon, not the income-convention atkinson().
    lorenz.lorenz_measures                   Z and Gamma; Z = 0 is Lorenz dominance (or coincident curves), as in visualize.py
    visualize.pairwise_disagreement          opposite non-tied signs on normalized differences, tie tolerance DEFAULT_TAU
    visualize._pair_rows                     the rows of A and B of every pair, by id

Layer 3 SD: the disagreement rate is computed for each c50 file, and the SD is the sample SD (ddof = 1) of those 100 rates, in
percentage points. The pooled rate is total disagreements / total observations, which equals the mean of the 100 rates.

Run with the Python used for the layer1 tools:
    python results/m1_reference/m1_reference.py               compute (if m1_reference_counts.npz is missing), outputs, checks
    python results/m1_reference/m1_reference.py --recompute   compute again from the samples
    python results/m1_reference/m1_reference.py --workers 3   number of processes for the computing step (default 4)
An interrupted computing step resumes from _checkpoints/, which is removed when it ends.
The Layer 3 denominator is checked separately, inside WSL, by check_layer3_denominator.py.
"""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from impact_function import linear_impact_function  # noqa: E402
from lorenz import DEFAULT_TOL as LORENZ_TOL, lorenz_measures  # noqa: E402
from metrics import _bad_atkinson_max, _bad_atkinson_rows, _gini_max, _gini_rows, _prepare_rows, _theil_max, beta_grid  # noqa: E402
from sampling import OUTCOMES  # noqa: E402
from visualize import DEFAULT_TAU, DISAGREEMENT, TIE, _pair_rows, pairwise_disagreement  # noqa: E402

FIGURES = HERE / "figures"
COUNTS = HERE / "m1_reference_counts.npz"
CHECKPOINTS = HERE / "_checkpoints"
L1_PATH = ROOT / "results" / "seed42_K52_n250000_b.parquet"
L2_PATH = ROOT / "results" / "seed42_K52_n250000_y.parquet"
L3_X_PATH = ROOT / "results" / "layer3" / "seed42_n25000_x.parquet"
L3_DIR = ROOT / "results" / "layer3" / "seed42_n25000_y"
SUMMARIES = ROOT / "results" / "m1"
TWIN_STEP = ROOT / "twin_not_included" / "burden_step.py"   # the digital twin is not part of this repository

REF_EPSILON, REF_MODE, REF_TIME = 1.0, 0.5, 0.5
PHI_STEP = 0.1
PHI_GRID = tuple(round(i * PHI_STEP, 10) for i in range(int(round(1 / PHI_STEP)) + 1))
SETTINGS = [(phi_mode, phi_time) for phi_mode in PHI_GRID for phi_time in PHI_GRID]   # the cross product
REF_SETTING = SETTINGS.index((REF_MODE, REF_TIME))
OLD_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # the grid of results/m1/, used for the comparison with it and as axis ticks
SHARED = tuple(v for v in OLD_GRID if v in PHI_GRID)
BETAS = np.sort(beta_grid())[::-1]   # -0.25, ..., -5: epsilon increasing
EPSILONS = -BETAS
REF_BETA = np.array([-REF_EPSILON])
REF_COLUMN = 2 + int(np.flatnonzero(EPSILONS == REF_EPSILON)[0])
GINI, THEIL = 0, 1                   # columns of the metric matrix; Atkinson columns start at 2
COMPARISONS = [("Gini vs Theil", "Gini", "Theil", np.nan, GINI, THEIL)]
COMPARISONS += [("Gini vs Atkinson", "Gini", "Atkinson", e, GINI, 2 + i) for i, e in enumerate(EPSILONS)]
COMPARISONS += [("Theil vs Atkinson", "Theil", "Atkinson", e, THEIL, 2 + i) for i, e in enumerate(EPSILONS)]
REF_NAMES = ("Gini vs Theil", "Gini vs Atkinson", "Theil vs Atkinson")
REF_Q = [0, 1 + REF_COLUMN - 2, 1 + len(EPSILONS) + REF_COLUMN - 2]
GRID_COLUMNS = ((GINI, THEIL), (GINI, 2), (THEIL, 2))   # in the matrix with one Atkinson column, epsilon = 1
EDGES = np.linspace(0, 0.5, 11)      # the M1 grid: 10 x 10 cells of width 0.05; Z and Gamma are at most 0.5 by definition
LORENZ_CHUNK = 10_000                # pairs per lorenz_measures call: its memory grows quickly with the chunk
SPARSE_BELOW = 50
LAYERS = ("Layer 1", "Layer 2", "Layer 3")


# ------------------------------------------------------------------ computing
def normalized_differences(b, rows_a, rows_b, p, betas=BETAS):
    """M(A) - M(B) of the normalized Gini, Theil and bad Atkinson (one column per beta) of every pair."""
    p, b, b_mean = _prepare_rows(p, b)
    r = b / b_mean[:, None]
    values = np.column_stack([_gini_rows(p, r), xlogy(r, r) @ p, _bad_atkinson_rows(p, r, betas)])
    normalized = values / np.concatenate([[_gini_max(p), _theil_max(p)], _bad_atkinson_max(p, betas)])
    return normalized[rows_a] - normalized[rows_b]


def disagreement_counts(delta):
    n_disagree, n_ties = np.empty(len(COMPARISONS), np.int64), np.empty(len(COMPARISONS), np.int64)
    flags = {}
    for q, (*_, j, k) in enumerate(COMPARISONS):
        status = pairwise_disagreement(delta[:, j], delta[:, k], DEFAULT_TAU)[1]
        flag = status == DISAGREEMENT
        n_disagree[q], n_ties[q] = flag.sum(), (status == TIE).sum()
        if q in REF_Q:
            flags[q] = flag
    return n_disagree, n_ties, flags


def lorenz_of(b, rows_a, rows_b, p):
    n = len(rows_a)
    z, gamma = np.empty(n), np.empty(n)
    for start in range(0, n, LORENZ_CHUNK):
        chunk = slice(start, start + LORENZ_CHUNK)
        z[chunk], gamma[chunk] = lorenz_measures(b[rows_a[chunk]], b[rows_b[chunk]], p)
    return z, gamma


def geometry(b, rows_a, rows_b, p, flags):
    """Counts of the heatmap at the reference. Cells include their lower edge; the last one also includes 0.5."""
    z, gamma = lorenz_of(b, rows_a, rows_b, p)
    dominance = z == 0
    cell = np.digitize(z, EDGES[1:-1]) * 10 + np.digitize(gamma, EDGES[1:-1])
    crossing = ~dominance
    out = {"n_dominance": int(dominance.sum()), "gamma_nonzero_under_dominance": int((gamma[dominance] != 0).sum()),
           "gamma_zero_when_crossing": int((gamma[crossing] == 0).sum()),
           "cell_crossing": np.bincount(cell[crossing], minlength=100).astype(np.int64),
           "cell_disagree": np.stack([np.bincount(cell[crossing & flags[q]], minlength=100) for q in REF_Q]).astype(np.int64),
           "disagree_under_dominance": np.array([int((flags[q] & dominance).sum()) for q in REF_Q], np.int64),
           "cell_index": np.where(crossing, cell, -1).astype(np.int8)}
    return out, z, gamma


def read_outcomes(path, group=None):
    pf = pq.ParquetFile(path)
    settings = json.loads(pf.schema_arrow.metadata[b"sampling"])
    names = [f"{outcome}_{k}" for outcome in OUTCOMES for k in range(1, settings["K"] + 1)]
    levers = [name for name in ("w_start", "w_end", "fee_base", "fee_step") if name in pf.schema_arrow.names]
    columns = ["id", *levers, *names]
    table = pf.read(columns=columns) if group is None else pf.read_row_group(group, columns=columns)
    rows_a, rows_b = _pair_rows(table["id"].to_numpy(), path.name)
    outcomes = tuple(np.column_stack([table[f"{outcome}_{k}"].to_numpy() for k in range(1, settings["K"] + 1)]) for outcome in OUTCOMES)
    x = np.column_stack([table[name].to_numpy() for name in levers]) if levers else None
    return outcomes, rows_a, rows_b, settings, x


def layer1_task(group):
    pf = pq.ParquetFile(L1_PATH)
    settings = json.loads(pf.schema_arrow.metadata[b"sampling"])
    K = settings["K"]
    p = np.asarray(settings["p"], float)
    atk = f"bad_Atk({-REF_EPSILON:g})_norm"
    table = pf.read_row_group(group, columns=["id", "zheng", "LDI", "GINI_norm", "Theil_norm", atk, *[f"b{k}" for k in range(1, K + 1)]])
    rows_a, rows_b = _pair_rows(table["id"].to_numpy(), L1_PATH.name)
    b = np.column_stack([table[f"b{k}"].to_numpy() for k in range(1, K + 1)])
    delta = normalized_differences(b, rows_a, rows_b, p)
    n_disagree, n_ties, flags = disagreement_counts(delta)
    out, z, gamma = geometry(b, rows_a, rows_b, p, flags)
    stored = lambda name: table[name].to_numpy()[rows_a] - table[name].to_numpy()[rows_b]
    out.update(layer="Layer 1", n=len(rows_a), n_disagree=n_disagree, n_ties=n_ties, stored_gap={
        "Z": float(np.abs(z - table["zheng"].to_numpy()[rows_a]).max()), "Gamma": float(np.abs(gamma - table["LDI"].to_numpy()[rows_a]).max()),
        "Gini": float(np.abs(delta[:, GINI] - stored("GINI_norm")).max()), "Theil": float(np.abs(delta[:, THEIL] - stored("Theil_norm")).max()),
        "Atkinson eps 1": float(np.abs(delta[:, REF_COLUMN] - stored(atk)).max())})
    return out


def grid_counts(y, rows_a, rows_b, p):
    """Disagreements and ties of the three comparisons at epsilon = 1, on the cross product of the phi grid."""
    n_disagree, n_ties = np.empty((len(SETTINGS), 3), np.int64), np.empty((len(SETTINGS), 3), np.int64)
    for i, (phi_mode, phi_time) in enumerate(SETTINGS):
        delta = normalized_differences(linear_impact_function(*y, w_mode=phi_mode, w_time=phi_time), rows_a, rows_b, p, REF_BETA)
        for r, (j, k) in enumerate(GRID_COLUMNS):
            status = pairwise_disagreement(delta[:, j], delta[:, k], DEFAULT_TAU)[1]
            n_disagree[i, r], n_ties[i, r] = (status == DISAGREEMENT).sum(), (status == TIE).sum()
    return n_disagree, n_ties


def phi_task(y, rows_a, rows_b, p):
    """All comparisons and the heatmap counts at the reference, and the phi grid at epsilon = 1."""
    b = linear_impact_function(*y, w_mode=REF_MODE, w_time=REF_TIME)
    n_disagree, n_ties, flags = disagreement_counts(normalized_differences(b, rows_a, rows_b, p))
    out = geometry(b, rows_a, rows_b, p, flags)[0]
    grid_disagree, grid_ties = grid_counts(y, rows_a, rows_b, p)
    out.update(n=len(rows_a), n_disagree=n_disagree, n_ties=n_ties, grid_disagree=grid_disagree, grid_ties=grid_ties)
    return out


def layer2_task(group):
    y, rows_a, rows_b, settings, _ = read_outcomes(L2_PATH, group)
    p = np.asarray(settings["p"], float)
    out = phi_task(y, rows_a, rows_b, p)
    # the columns stored in the file are at the weights in its metadata: recompute those for a comparison
    pf = pq.ParquetFile(L2_PATH)
    impact = json.loads(pf.schema_arrow.metadata[b"impact"])
    atk = f"bad_Atk({-REF_EPSILON:g})_norm"
    table = pf.read_row_group(group, columns=["zheng", "LDI", "GINI_norm", "Theil_norm", atk])
    b = linear_impact_function(*y, w_mode=impact["w_mode"], w_time=impact["w_time"])
    delta = normalized_differences(b, rows_a, rows_b, p)
    z, gamma = lorenz_of(b, rows_a, rows_b, p)
    stored = lambda name: table[name].to_numpy()[rows_a] - table[name].to_numpy()[rows_b]
    out.update(layer="Layer 2", stored_gap={
        "Z": float(np.abs(z - table["zheng"].to_numpy()[rows_a]).max()), "Gamma": float(np.abs(gamma - table["LDI"].to_numpy()[rows_a]).max()),
        "Gini": float(np.abs(delta[:, GINI] - stored("GINI_norm")).max()), "Theil": float(np.abs(delta[:, THEIL] - stored("Theil_norm")).max()),
        "Atkinson eps 1": float(np.abs(delta[:, REF_COLUMN] - stored(atk)).max())})
    return out


def layer3_task(path):
    y, rows_a, rows_b, settings, x = read_outcomes(Path(path))
    out = phi_task(y, rows_a, rows_b, np.asarray(settings["p"], float))
    out.update(layer="Layer 3", c50=float(settings["c50"]), file=Path(path).name, levers_hash=hashlib.sha256(x.tobytes()).hexdigest(),
               baseline_hash=hashlib.sha256(np.asarray(settings["baseline_trips"]).tobytes()).hexdigest(),
               columns=settings["columns"], baseline_total=float(np.sum(settings["baseline_trips"])))
    return out


def run_task(task):
    """One task, with a checkpoint: an interrupted run resumes where it stopped."""
    name, kind, argument = task
    target = CHECKPOINTS / f"{name}.pkl"
    if target.exists():
        return name, 0.0
    t_start = time.perf_counter()
    result = {"L1": layer1_task, "L2": layer2_task, "L3": layer3_task}[kind](argument)
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(result))
    os.replace(temporary, target)
    return name, time.perf_counter() - t_start


def compute(workers):
    t_start = time.perf_counter()
    CHECKPOINTS.mkdir(exist_ok=True)
    groups = {path: pq.ParquetFile(path).metadata.num_row_groups for path in (L1_PATH, L2_PATH)}
    tasks = [(f"L1_{g}", "L1", g) for g in range(groups[L1_PATH])] + [(f"L2_{g}", "L2", g) for g in range(groups[L2_PATH])]
    tasks += [(f"L3_{path.stem}", "L3", str(path)) for path in sorted(L3_DIR.glob("*_y.parquet"))]
    print(f"{len(tasks)} tasks, {len(SETTINGS)} phi settings, {workers} processes", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (name, seconds) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            if i <= 12 or i % 5 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done ({name}: {seconds:.0f} s), total {time.perf_counter() - t_start:.0f} s", flush=True)
    results = [pickle.loads((CHECKPOINTS / f"{task[0]}.pkl").read_bytes()) for task in tasks]
    data, text = {}, {}
    for layer, key in zip(LAYERS, ("L1", "L2", "L3")):
        parts = [r for r in results if r["layer"] == layer]
        if layer == "Layer 3":
            parts.sort(key=lambda r: r["c50"])
            data["L3_c50"] = np.array([r["c50"] for r in parts])
            indices = np.stack([r["cell_index"] for r in parts])
            data["L3_cell_distinct"] = np.array([(indices == c).any(axis=0).sum() for c in range(100)], np.int64)
            text["L3"] = {"files": [r["file"] for r in parts], "levers_hashes": sorted({r["levers_hash"] for r in parts}),
                          "baseline_hashes": sorted({r["baseline_hash"] for r in parts}), "columns": sorted({r["columns"] for r in parts}),
                          "baseline_total": parts[0]["baseline_total"]}
        else:
            text[key] = {"stored_gap": {name: max(r["stored_gap"][name] for r in parts) for name in parts[0]["stored_gap"]}}
        data[f"{key}_n"] = np.array([r["n"] for r in parts], np.int64)
        for name in ("n_dominance", "gamma_nonzero_under_dominance", "gamma_zero_when_crossing"):
            data[f"{key}_{name}"] = np.array([r[name] for r in parts], np.int64)
        for name in ("n_disagree", "n_ties", "cell_crossing", "cell_disagree", "disagree_under_dominance"):
            data[f"{key}_{name}"] = np.stack([r[name] for r in parts])
        if layer != "Layer 1":
            for name in ("grid_disagree", "grid_ties"):
                data[f"{key}_{name}"] = np.stack([r[name] for r in parts])
    np.savez_compressed(COUNTS, text=json.dumps(text), phi_grid=np.array(PHI_GRID), compute_seconds=time.perf_counter() - t_start, **data)
    shutil.rmtree(CHECKPOINTS)
    print(f"saved {COUNTS.name}, {time.perf_counter() - t_start:.0f} s", flush=True)


# ------------------------------------------------------------------ results from the counts
class Results:
    """Rates and cells of the three layers, from m1_reference_counts.npz."""

    def __init__(self):
        with np.load(COUNTS, allow_pickle=False) as npz:
            self.data = {key: npz[key] for key in npz.files}
        if not np.allclose(self.data["phi_grid"], PHI_GRID) or len(self.data["phi_grid"]) != len(PHI_GRID):
            raise ValueError(f"{COUNTS.name} was computed on another phi grid: run with --recompute")
        self.text = json.loads(str(self.data["text"]))
        self.c50 = self.data["L3_c50"]
        keys = {"Layer 1": "L1", "Layer 2": "L2", "Layer 3": "L3"}
        self.n = {layer: int(self.data[f"{key}_n"].sum()) for layer, key in keys.items()}
        self.disagree = {layer: self.data[f"{key}_n_disagree"].sum(0) for layer, key in keys.items()}
        self.ties = {layer: self.data[f"{key}_n_ties"].sum(0) for layer, key in keys.items()}
        self.grid_disagree = {layer: self.data[f"{keys[layer]}_grid_disagree"].sum(0) for layer in ("Layer 2", "Layer 3")}
        self.grid_ties = {layer: self.data[f"{keys[layer]}_grid_ties"].sum(0) for layer in ("Layer 2", "Layer 3")}
        n = self.data["L3_n"]
        self.per_c50 = 100 * self.data["L3_n_disagree"] / n[:, None]                     # c50, comparison
        self.grid_per_c50 = 100 * self.data["L3_grid_disagree"] / n[:, None, None]       # c50, setting, comparison of the three
        self.sd_all = self.per_c50.std(axis=0, ddof=1)
        self.grid_sd_all = self.grid_per_c50.std(axis=0, ddof=1)

    def key(self, layer):
        return {"Layer 1": "L1", "Layer 2": "L2", "Layer 3": "L3"}[layer]

    def rate(self, layer, q):
        """Rate at the reference profile; q is one of the 41 comparisons."""
        return 100 * self.disagree[layer][q] / self.n[layer]

    def sd(self, q):
        return self.sd_all[q]

    def grid_rate(self, layer, setting, r):
        """Rate at epsilon = 1 at one setting of the phi grid; r = 0, 1, 2 as in REF_NAMES."""
        return 100 * self.grid_disagree[layer][setting, r] / self.n[layer]

    def grid_sd(self, setting, r):
        return self.grid_sd_all[setting, r]

    def cells(self, layer):
        key = self.key(layer)
        crossing, disagree = self.data[f"{key}_cell_crossing"].sum(0), self.data[f"{key}_cell_disagree"].sum(0)
        distinct = self.data["L3_cell_distinct"] if layer == "Layer 3" else crossing
        return crossing, disagree, distinct

    def crossing_rate(self, layer):
        key = self.key(layer)
        return 100 * (1 - self.data[f"{key}_n_dominance"].sum() / self.n[layer])


def long_rates(res):
    """All computed rates: the reference with the full epsilon grid, then the other phi settings at epsilon = 1."""
    rows = []

    def layer3_columns(values, sd):
        return {"c50_realizations": len(res.c50), "distinct_policy_pairs": int(res.data["L3_n"][0]), "sd_across_c50_pct_points": sd,
                "min_across_c50_pct": values.min(), "max_across_c50_pct": values.max()}
    for layer in LAYERS:
        reference = (None, None) if layer == "Layer 1" else (REF_MODE, REF_TIME)
        for q, (name, a, b, epsilon, *_) in enumerate(COMPARISONS):
            row = {"layer": layer, "phi_mode": reference[0], "phi_time": reference[1], "comparison": name, "metric_a": a, "metric_b": b, "epsilon": epsilon,
                   "n_pairs": res.n[layer], "n_disagree": int(res.disagree[layer][q]), "n_ties": int(res.ties[layer][q]), "disagree_pct_of_all_pairs": res.rate(layer, q)}
            if layer == "Layer 3":
                row.update(layer3_columns(res.per_c50[:, q], res.sd(q)))
            rows.append(row)
        if layer == "Layer 1":
            continue
        for s, (phi_mode, phi_time) in enumerate(SETTINGS):
            if s == REF_SETTING:
                continue
            for r, (name, q) in enumerate(zip(REF_NAMES, REF_Q)):
                _, a, b, epsilon, *_ = COMPARISONS[q]
                row = {"layer": layer, "phi_mode": phi_mode, "phi_time": phi_time, "comparison": name, "metric_a": a, "metric_b": b, "epsilon": epsilon,
                       "n_pairs": res.n[layer], "n_disagree": int(res.grid_disagree[layer][s, r]), "n_ties": int(res.grid_ties[layer][s, r]), "disagree_pct_of_all_pairs": res.grid_rate(layer, s, r)}
                if layer == "Layer 3":
                    row.update(layer3_columns(res.grid_per_c50[:, s, r], res.grid_sd(s, r)))
                rows.append(row)
    return pd.DataFrame(rows)


def write_table(res):
    rows = []
    for name, q in zip(REF_NAMES, REF_Q):
        rows.append({"comparison": name, "atkinson_epsilon": "" if np.isnan(COMPARISONS[q][3]) else f"{COMPARISONS[q][3]:g}",
                     "layer_1_pct": res.rate("Layer 1", q), "layer_2_pct": res.rate("Layer 2", q),
                     "layer_3_pooled_pct": res.rate("Layer 3", q), "layer_3_sd_across_c50_pct_points": res.sd(q)})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "m1_reference_table.csv", index=False, float_format="%.6f")
    labels = {"Gini vs Theil": r"Gini vs Theil", "Gini vs Atkinson": r"Gini vs Atkinson, $\varepsilon=1$", "Theil vs Atkinson": r"Theil vs Atkinson, $\varepsilon=1$"}
    lines = [r"% Requires \usepackage{booktabs}. Generated by m1_reference.py.", r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Pairwise metric disagreement at an illustrative stakeholder reference profile, which is not a universal or empirically elicited "
             r"preference: Atkinson $\varepsilon=1$ and, in Layers~2 and~3, $\varphi_{\mathrm{mode}}=\varphi_{\mathrm{time}}=0.5$. Layer~1 has no $\varphi$ parameters. "
             rf"Entries are percentages of all sampled pairs: {res.n['Layer 1']:,} pairs in Layers~1 and~2, and {int(res.data['L3_n'][0]):,} policy pairs evaluated at "
             rf"{len(res.c50)} $c_{{50}}$ realizations in Layer~3. Layer~3 entries give the pooled rate, with the standard deviation across $c_{{50}}$ realizations in "
             r"parentheses; it describes variation across realizations and is not a confidence interval.}",
             r"\label{tab:m1_reference_pairwise}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Comparison & Layer~1 & Layer~2 & Layer~3 pooled (SD across $c_{50}$) \\", r"\midrule"]
    for row in rows:
        lines.append(f"{labels[row['comparison']]} & {row['layer_1_pct']:.2f} & {row['layer_2_pct']:.2f} & {row['layer_3_pooled_pct']:.2f} ({row['layer_3_sd_across_c50_pct_points']:.2f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "m1_reference_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def write_cells(res):
    rows = []
    for layer in LAYERS:
        crossing, disagree, distinct = res.cells(layer)
        for c in range(100):
            for r, (name, q) in enumerate(zip(REF_NAMES, REF_Q)):
                rows.append({"layer": layer, "phi_mode": "" if layer == "Layer 1" else REF_MODE, "phi_time": "" if layer == "Layer 1" else REF_TIME, "comparison": name,
                             "atkinson_epsilon": "" if np.isnan(COMPARISONS[q][3]) else f"{COMPARISONS[q][3]:g}", "z_low": EDGES[c // 10], "z_high": EDGES[c // 10 + 1],
                             "gamma_low": EDGES[c % 10], "gamma_high": EDGES[c % 10 + 1], "n_nondominance_observations": int(crossing[c]),
                             "n_nondominance_distinct_pairs": int(distinct[c]), "n_disagree": int(disagree[r, c]),
                             "disagree_pct_excl_dominance": 100 * disagree[r, c] / crossing[c] if crossing[c] else np.nan,
                             "empty": bool(crossing[c] == 0), "sparse": bool(0 < distinct[c] < SPARSE_BELOW)})
    cells = pd.DataFrame(rows)
    cells.to_csv(HERE / "m1_reference_heatmap_cells.csv", index=False, float_format="%.6f")
    return cells


def write_per_c50(res):
    rows = []
    for c, value in enumerate(res.c50):
        for s, (phi_mode, phi_time) in enumerate(SETTINGS):
            for r, name in enumerate(REF_NAMES):
                rows.append((value, phi_mode, phi_time, name, "" if r == 0 else f"{REF_EPSILON:g}", int(res.data["L3_n"][c]), int(res.data["L3_grid_disagree"][c, s, r]), res.grid_per_c50[c, s, r]))
    columns = ("c50", "phi_mode", "phi_time", "comparison", "atkinson_epsilon", "n_pairs", "n_disagree", "disagree_pct")
    pd.DataFrame(rows, columns=columns).to_csv(HERE / "m1_reference_layer3_per_c50.csv", index=False, float_format="%.6f")


# ------------------------------------------------------------------ figures
STYLE = {"Gini vs Theil": ("#2a78d6", "o"), "Gini vs Atkinson": ("#eb6834", "s"), "Theil vs Atkinson": ("#1baf7a", "^")}
INK, INK2, MUTED, HAIR, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = ("#e9f2fc", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
MODE, TIME = r"$\varphi_{\mathrm{mode}}$", r"$\varphi_{\mathrm{time}}$"
EPS_MATH = r"$\varepsilon$"


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": HAIR, "grid.linewidth": .6, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", pad_inches=.03)


def draw(ax, layer, name, x, y, sd=None, marker=True, capsize=0.0):
    color, shape = STYLE[name]
    ax.plot(x, y, color=color, lw=1.4, marker=shape if marker else None, ms=3.2, mec="white", mew=.35, label=name, gid=f"{layer}|{name}", zorder=3)
    if sd is not None:
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=.2, lw=0, zorder=1)
        bars = ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=color, elinewidth=.7, capsize=capsize, capthick=.7, zorder=2)
        bars.lines[2][0].set_gid(f"{layer}|{name}|sd")


def legend_above(fig, ax, y=1.0, with_sd=True):
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    if with_sd:
        handles.append(Patch(facecolor=MUTED, alpha=.35, edgecolor="none"))
        labels.append(r"Layer 3: $\pm$1 SD across 100 $c_{50}$ realizations")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(.5, y), handlelength=1.8, columnspacing=1.4)


def figure_epsilon(res):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True, constrained_layout=True)
    titles = {"Layer 1": "Layer 1 (no $\\varphi$ parameters)", "Layer 2": f"Layer 2 ({MODE} = {TIME} = 0.5)", "Layer 3": f"Layer 3 ({MODE} = {TIME} = 0.5)"}
    for ax, layer in zip(axes, LAYERS):
        for name, first in (("Gini vs Theil", 0), ("Gini vs Atkinson", 1), ("Theil vs Atkinson", 1 + len(EPSILONS))):
            qs = [0] * len(EPSILONS) if name == "Gini vs Theil" else list(range(first, first + len(EPSILONS)))
            y = np.array([res.rate(layer, q) for q in qs])
            sd = np.array([res.sd(q) for q in qs]) if layer == "Layer 3" else None
            draw(ax, layer, name, EPSILONS, y, sd, marker=name != "Gini vs Theil")
        ax.axvline(REF_EPSILON, color=MUTED, lw=.7, zorder=0)
        ax.set_title(titles[layer], loc="left")
        ax.set_xlabel(f"Atkinson {EPS_MATH}")
        ax.set_xlim(0, 5.15)
        ax.set_xticks([.25, 1, 2, 3, 4, 5], ["0.25", "1", "2", "3", "4", "5"])
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Pairs disagreeing, %")
    legend_above(fig, axes[2], 1.0)
    save(fig, "m1_reference_pairwise_vs_epsilon")
    return fig


def figure_phi(res, varying):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True, constrained_layout=True)
    indices = [SETTINGS.index((v, REF_TIME) if varying == "phi_mode" else (REF_MODE, v)) for v in PHI_GRID]
    x = np.array(PHI_GRID)
    fixed = f"{TIME} = 0.5" if varying == "phi_mode" else f"{MODE} = 0.5"
    for ax, layer in zip(axes, ("Layer 2", "Layer 3")):
        for r, name in enumerate(REF_NAMES):
            y = np.array([res.grid_rate(layer, s, r) for s in indices])
            sd = np.array([res.grid_sd(s, r) for s in indices]) if layer == "Layer 3" else None
            draw(ax, layer, name, x, y, sd, capsize=1.4)
        ax.axvline(.5, color=MUTED, lw=.7, zorder=0)
        ax.set_title(f"{layer} ({EPS_MATH} = 1, {fixed})", loc="left")
        ax.set_xlabel(MODE if varying == "phi_mode" else TIME)
        ax.set_xticks(OLD_GRID, [f"{v:g}" for v in OLD_GRID])
        ax.set_xlim(-.03, 1.03)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Pairs disagreeing, %")
    legend_above(fig, axes[1], 1.0)
    save(fig, f"m1_reference_pairwise_vs_{varying}")
    return fig


def figure_surface(res):
    """The whole cross product at epsilon = 1, one panel per layer and comparison, on one color scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    surfaces = {(layer, r): res.grid_disagree[layer][:, r].reshape(len(PHI_GRID), len(PHI_GRID)) * 100 / res.n[layer] for layer in ("Layer 2", "Layer 3") for r in range(3)}
    top = 2 * np.ceil(max(v.max() for v in surfaces.values()) / 2)
    norm = Normalize(0, top)
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    edges = np.concatenate([[-PHI_STEP / 2], np.array(PHI_GRID) + PHI_STEP / 2])
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), sharex=True, sharey=True, constrained_layout=True)
    labels = {"Gini vs Theil": "Gini vs Theil", "Gini vs Atkinson": f"Gini vs Atkinson ({EPS_MATH} = 1)", "Theil vs Atkinson": f"Theil vs Atkinson ({EPS_MATH} = 1)"}
    for i, layer in enumerate(("Layer 2", "Layer 3")):
        for r, name in enumerate(REF_NAMES):
            ax = axes[i, r]
            mesh = ax.pcolormesh(edges, edges, surfaces[(layer, r)], cmap=cmap, norm=norm, edgecolors="face", linewidth=.1)
            ax.scatter([REF_TIME], [REF_MODE], s=14, color=INK, edgecolors="white", linewidths=.5, zorder=4)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xticks(OLD_GRID, [f"{v:g}" for v in OLD_GRID])
            ax.set_yticks(OLD_GRID, [f"{v:g}" for v in OLD_GRID])
            if i == 0:
                ax.set_title(labels[name], loc="left")
            if i == 1:
                ax.set_xlabel(TIME)
            if r == 0:
                ax.set_ylabel(f"{layer}\n{MODE}")
    bar = fig.colorbar(mesh, ax=axes, shrink=.8, pad=.015, aspect=30)
    bar.set_label("Pairs disagreeing, %")
    bar.outline.set_visible(False)
    save(fig, "m1_reference_phi_surface")
    return fig, float(top)


def color_scale(res):
    """Top of the common scale: the highest rate among non-sparse cells, rounded up to a multiple of 5."""
    top, above = 0.0, 0
    for layer in LAYERS:
        crossing, disagree, distinct = res.cells(layer)
        rates = 100 * disagree / np.where(crossing > 0, crossing, 1)
        solid = (crossing > 0) & (distinct >= SPARSE_BELOW)
        top = max(top, float(rates[:, solid].max()))
    top = 5 * np.ceil(top / 5)
    for layer in LAYERS:
        crossing, disagree, _ = res.cells(layer)
        above += int((100 * disagree[:, crossing > 0] / crossing[crossing > 0] > top).sum())
    return float(top), above


def figure_heatmap(res):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    top, above = color_scale(res)
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)
    cmap.set_bad("white", 0)
    norm = Normalize(0, top)
    fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.35), sharex=True, sharey=True, constrained_layout=True)
    heads = {"Layer 1": "Layer 1 (no $\\varphi$ parameters)", "Layer 2": f"Layer 2 ({MODE} = {TIME} = 0.5)", "Layer 3": f"Layer 3 ({MODE} = {TIME} = 0.5)"}
    rows = {"Gini vs Theil": "Gini vs Theil", "Gini vs Atkinson": f"Gini vs Atkinson ({EPS_MATH} = 1)", "Theil vs Atkinson": f"Theil vs Atkinson ({EPS_MATH} = 1)"}
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    for i, (name, q) in enumerate(zip(REF_NAMES, REF_Q)):
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j]
            crossing, disagree, distinct = res.cells(layer)
            grid = np.full((10, 10), np.nan)    # rows Gamma, columns Z
            for c in np.flatnonzero(crossing):
                grid[c % 10, c // 10] = 100 * disagree[i, c] / crossing[c]
            mesh = ax.pcolormesh(EDGES, EDGES, np.ma.masked_invalid(grid), cmap=cmap, norm=norm, edgecolors="white", linewidth=.4, rasterized=False)
            sparse = np.flatnonzero((distinct > 0) & (distinct < SPARSE_BELOW))
            if len(sparse):
                ax.scatter(centers[sparse // 10], centers[sparse % 10], s=5, color=INK, edgecolors="white", linewidths=.3, zorder=4)
            ax.set_title((heads[layer] + "\n" if i == 0 else "") + f"{res.rate(layer, q):.2f}% of all pairs", loc="left", fontsize=8)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xlim(0, .5)
            ax.set_ylim(0, .5)
            ax.set_xticks([0, .25, .5], ["0", "0.25", "0.5"])
            ax.set_yticks([0, .25, .5], ["0", "0.25", "0.5"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(True)
            if i == 2:
                ax.set_xlabel("Zheng descriptor $Z$")
            if j == 0:
                ax.set_ylabel(f"{rows[name]}\n" + r"Lorenz-intersection measure $\Gamma$")
    bar = fig.colorbar(mesh, ax=axes, shrink=.72, pad=.015, aspect=34, extend="max" if above else "neither", extendfrac=.025)
    bar.set_label("Disagreement among non-dominance pairs in the cell, %")
    bar.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=2.6, color=INK, mec="white", mew=.3)], labels=[f"Fewer than {SPARSE_BELOW} non-dominance pairs"],
               loc="upper center", bbox_to_anchor=(.47, 0), frameon=False, handletextpad=.3)
    save(fig, "m1_reference_heatmap")
    return fig, top, above


# ------------------------------------------------------------------ checks
class Checks:
    def __init__(self):
        self.lines, self.failed = [], 0

    def add(self, ok, text):
        self.failed += not ok
        self.lines.append(("ok   " if ok else "DISCREPANCY ") + text)
        print(self.lines[-1], flush=True)


def plotted(fig, gid):
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_gid() == gid:
                return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
    raise KeyError(gid)


def plotted_sd(fig, gid):
    for ax in fig.axes:
        for collection in ax.collections:
            if collection.get_gid() == gid:
                return np.array([abs(segment[1, 1] - segment[0, 1]) / 2 for segment in collection.get_segments()])
    raise KeyError(gid)


def run_checks(res, cells, figures, top, above):
    check = Checks()
    # samples
    for layer in LAYERS:
        key = res.key(layer)
        crossing, disagree, _ = res.cells(layer)
        n_dom = int(res.data[f"{key}_n_dominance"].sum())
        check.add(crossing.sum() + n_dom == res.n[layer], f"{layer}: non-dominance cell counts + dominance pairs = all pairs ({res.n[layer]:,})")
        check.add((disagree.sum(1) == res.disagree[layer][REF_Q]).all(), f"{layer}: cell disagreements add up to the table's numerators; so dominance pairs contribute none")
        check.add(int(res.data[f"{key}_disagree_under_dominance"].max()) == 0, f"{layer}: no disagreement among Lorenz-dominance pairs in the three comparisons")
        check.add(int(res.data[f"{key}_gamma_nonzero_under_dominance"].sum()) == 0 and int(res.data[f"{key}_gamma_zero_when_crossing"].sum()) == 0,
                  f"{layer}: Gamma = 0 exactly when Z = 0")
    check.add(res.n["Layer 1"] == res.n["Layer 2"] == 250_000, "Layers 1 and 2 hold 250,000 pairs each")
    check.add(len(res.c50) == 100 and len(np.unique(res.c50)) == 100 and bool((res.data["L3_n"] == 25_000).all()), f"Layer 3 has 100 distinct c50 realizations of 25,000 pairs each (c50 from {res.c50.min():.4f} to {res.c50.max():.4f})")
    check.add(len(res.text["L3"]["levers_hashes"]) == 1, "Layer 3: the same policy pairs in all 100 files (identical levers)")
    ps = {name: np.asarray(json.loads(pq.read_schema(path).metadata[b"sampling"])["p"]) for name, path in (("L1", L1_PATH), ("L2", L2_PATH), ("L3", sorted(L3_DIR.glob("*_y.parquet"))[0]))}
    gap = max(np.abs(ps["L1"] - ps["L3"]).max(), np.abs(ps["L2"] - ps["L3"]).max())
    check.add(gap < 1e-12, f"the three layers use the same K = {len(ps['L1'])} and p (largest difference {gap:.1e})")
    file_betas = {name: sorted(json.loads(pq.read_schema(path).metadata[b"metrics"])["betas"]) for name, path in (("L1", L1_PATH), ("L2", L2_PATH))}
    check.add(all(np.allclose(v, np.sort(BETAS)) for v in file_betas.values()), f"the epsilon grid ({EPSILONS[0]:g}, ..., {EPSILONS[-1]:g}, {len(EPSILONS)} values) is the grid stored in the Layer 1 and Layer 2 files")
    for key, layer in (("L1", "Layer 1"), ("L2", "Layer 2 at the weights of its stored columns (0.5, 0.25)")):
        g = res.text[key]["stored_gap"]
        check.add(max(g.values()) < 1e-9, f"{layer}: recomputed metrics, Z and Gamma vs the columns stored in the file: largest differences " + ", ".join(f"{k} {v:.1e}" for k, v in g.items()))
    # the phi grid
    check.add(len(SETTINGS) == len(set(SETTINGS)) == len(PHI_GRID) ** 2 and PHI_GRID[0] == 0 and PHI_GRID[-1] == 1 and (REF_MODE, REF_TIME) in SETTINGS, f"phi grid: {len(PHI_GRID)} values per parameter from 0 to 1, step {PHI_STEP:g}, {len(SETTINGS)} settings in the cross product, all distinct, including the reference")
    for layer in ("Layer 2", "Layer 3"):
        check.add((res.grid_disagree[layer][REF_SETTING] == res.disagree[layer][REF_Q]).all() and (res.grid_ties[layer][REF_SETTING] == res.ties[layer][REF_Q]).all(), f"{layer}: the phi grid at (0.5, 0.5) gives the same counts as the full computation at the reference")
    zero = SETTINGS.index((0.0, 0.0))
    same_counts = max(int(np.ptp(res.data["L3_grid_disagree"][:, zero], axis=0).max()), int(np.ptp(res.data["L3_grid_ties"][:, zero], axis=0).max())) == 0
    check.add(same_counts and float(res.grid_sd_all[zero].max()) < 1e-10, "Layer 3: at phi = (0, 0) the counts are the same at all 100 c50 values, as the twin's formula implies (P_r factors out of the cancelled share), so the SD is 0 up to floating-point rounding")
    check.add(float(res.grid_sd_all[[i for i in range(len(SETTINGS)) if i != zero]].min()) > 0, "Layer 3: SD across c50 is positive at every other phi setting")
    # denominator
    check.add(len(res.text["L3"]["baseline_hashes"]) == 1, "Layer 3: one baseline vector for all policies and c50 values, so it cannot depend on the policy's window")
    check.add(res.text["L3"]["columns"] == ["w_start ... fee_step: the policy; <outcome>_<group>: share of the group's daily baseline AV-bound trips with that outcome; the rest are rigid or exempted"], "Layer 3: the file metadata describes the shares as relative to daily baseline trips")
    denominator = json.loads((HERE / "denominator_check.json").read_text(encoding="utf-8"))
    check.add(denominator["conclusion"] == "whole-day", f"Layer 3 shares recomputed in WSL with the twin's inflow ({denominator['policies_checked']} policies x 3 c50): largest gap to the stored values {denominator['largest_abs_gap_stored_vs_whole_day_denominator']:.1e} with the whole-day denominator, {denominator['largest_abs_gap_stored_vs_window_denominator']:.3f} with the window denominator")
    # pooled rate and SD
    pooled = np.array([res.rate("Layer 3", q) for q in range(len(COMPARISONS))])
    grid_pooled = np.array([[res.grid_rate("Layer 3", s, r) for r in range(3)] for s in range(len(SETTINGS))])
    check.add(max(float(np.abs(pooled - res.per_c50.mean(0)).max()), float(np.abs(grid_pooled - res.grid_per_c50.mean(0)).max())) < 1e-10, "Layer 3: the pooled rate equals the mean of the 100 per-c50 rates, at the reference and on the phi grid")
    # the README's statement about the reference epsilon
    lowest = {layer: EPSILONS[int(np.argmin([res.rate(layer, q) for q in range(1, 1 + len(EPSILONS))]))] for layer in LAYERS}
    check.add(lowest["Layer 1"] == REF_EPSILON and lowest["Layer 2"] == REF_EPSILON, f"epsilon with the lowest Gini vs Atkinson rate at the reference: Layer 1 {lowest['Layer 1']:g}, Layer 2 {lowest['Layer 2']:g}, Layer 3 {lowest['Layer 3']:g}")
    # table on disk against the plotted points
    disk = pd.read_csv(HERE / "m1_reference_table.csv")
    worst = 0.0
    for name, q in zip(REF_NAMES, REF_Q):
        row = disk[disk.comparison == name].iloc[0]
        values = {"Layer 1": row.layer_1_pct, "Layer 2": row.layer_2_pct, "Layer 3": row.layer_3_pooled_pct}
        for layer, value in values.items():
            x, y = plotted(figures["epsilon"], f"{layer}|{name}")
            worst = max(worst, abs(y[np.flatnonzero(x == REF_EPSILON)[0]] - value))
            if layer != "Layer 1":
                for varying in ("phi_mode", "phi_time"):
                    x, y = plotted(figures[varying], f"{layer}|{name}")
                    worst = max(worst, abs(y[np.flatnonzero(x == .5)[0]] - value))
        sd = row.layer_3_sd_across_c50_pct_points
        x, _ = plotted(figures["epsilon"], f"Layer 3|{name}")
        worst = max(worst, abs(plotted_sd(figures["epsilon"], f"Layer 3|{name}|sd")[np.flatnonzero(x == REF_EPSILON)[0]] - sd))
        for varying in ("phi_mode", "phi_time"):
            x, _ = plotted(figures[varying], f"Layer 3|{name}")
            worst = max(worst, abs(plotted_sd(figures[varying], f"Layer 3|{name}|sd")[np.flatnonzero(x == .5)[0]] - sd))
    check.add(worst < 1e-5, f"the table on disk matches the reference points and their SD bars read back from the three line figures (largest difference {worst:.1e}, the CSV's rounding)")
    n_bars = {varying: len(plotted_sd(figures[varying], f"Layer 3|{REF_NAMES[0]}|sd")) for varying in ("phi_mode", "phi_time")}
    check.add(all(v == len(PHI_GRID) for v in n_bars.values()), f"the two phi figures show an SD bar at each of the {len(PHI_GRID)} plotted Layer 3 points")
    for name, q in zip(REF_NAMES, REF_Q):
        for layer in LAYERS:
            q_cells = cells[(cells.layer == layer) & (cells.comparison == name)]
            value = 100 * q_cells.n_disagree.sum() / res.n[layer]
            check.add(abs(value - res.rate(layer, q)) < 1e-10, f"{layer}, {name}: the heatmap cells give back the table's rate ({value:.2f}%)")
    # the existing summaries, on the settings they contain
    old = pd.read_csv(SUMMARIES / "m1_overall_rates.csv")
    new = long_rates(res)
    worst_rate, worst_count, worst_sd, missing, compared = 0.0, 0, 0.0, 0, 0
    for row in new.itertuples(index=False):
        if row.layer != "Layer 1" and (row.phi_mode not in SHARED or row.phi_time not in SHARED):
            continue
        q = old[(old.layer == row.layer) & (old.comparison == row.comparison)]
        if row.layer != "Layer 1":
            q = q[np.isclose(q.phi_mode, row.phi_mode) & np.isclose(q.phi_time, row.phi_time)]
        q = q[q.epsilon.isna()] if np.isnan(row.epsilon) else q[np.isclose(q.epsilon, row.epsilon)]
        if len(q) != 1:
            missing += 1
            continue
        q = q.iloc[0]
        compared += 1
        worst_rate = max(worst_rate, abs(q.disagree_pct - row.disagree_pct_of_all_pairs))
        worst_count = max(worst_count, abs(int(q.n_disagree) - row.n_disagree), abs(int(q.n_ties) - row.n_ties), abs(int(q.n_pairs) - row.n_pairs))
        if row.layer == "Layer 3":
            worst_sd = max(worst_sd, abs(q.epistemic_sd_pct - row.sd_across_c50_pct_points))
    check.add(missing == 0 and worst_count == 0 and worst_rate < 1e-5 and worst_sd < 1e-5, f"results/m1/m1_overall_rates.csv vs this recomputation, {compared:,} rows on the settings it contains: largest difference in counts {worst_count}, in rates {worst_rate:.1e}, in SD {worst_sd:.1e}; rows not found {missing}")
    context = pd.read_csv(SUMMARIES / "m1_context_rates.csv")
    worst = 0
    for layer in LAYERS:
        q = context[context.layer == layer]
        if layer != "Layer 1":
            q = q[np.isclose(q.phi_mode, REF_MODE) & np.isclose(q.phi_time, REF_TIME)]
        worst = max(worst, abs(int(q.n_lorenz_dominance.iloc[0]) - int(res.data[f"{res.key(layer)}_n_dominance"].sum())))
    check.add(worst == 0, "results/m1/m1_context_rates.csv: the same number of dominance pairs at the reference")
    worst_n, worst_d = 0, 0
    for layer, filename in zip(LAYERS, ("m1_cells_layer1.csv", "m1_cells_layer2.csv", "m1_cells_layer3.csv")):
        old_cells = pd.read_csv(SUMMARIES / filename)
        if layer != "Layer 1":
            old_cells = old_cells[np.isclose(old_cells.phi_mode, REF_MODE) & np.isclose(old_cells.phi_time, REF_TIME)]
        crossing, disagree, _ = res.cells(layer)
        for r, (name, q) in enumerate(zip(REF_NAMES, REF_Q)):
            part = old_cells[old_cells.comparison == name]
            part = part[part.epsilon.isna()] if np.isnan(COMPARISONS[q][3]) else part[np.isclose(part.epsilon, REF_EPSILON)]
            for row in part.itertuples(index=False):
                c = int(round(row.z_low / .05)) * 10 + int(round(row.gamma_low / .05))
                worst_n = max(worst_n, abs(int(row.n_pairs) - int(row.n_dominance) - int(crossing[c])))
                worst_d = max(worst_d, abs(int(row.n_disagree) - int(disagree[r, c])))
    check.add(worst_n == 0 and worst_d == 0, f"results/m1/m1_cells_layer*.csv at the reference: largest difference in non-dominance counts {worst_n}, in disagreements {worst_d}")
    other = SUMMARIES / "m1_reference_pairwise_eps1_phi0.5_0.5.csv"
    if other.exists():
        q = pd.read_csv(other)
        same = True
        for row, mine in zip(q.itertuples(index=False), disk.itertuples(index=False)):
            same &= f"{row[1]:.2f}" == f"{mine.layer_1_pct:.2f}" and f"{row[2]:.2f}" == f"{mine.layer_2_pct:.2f}"
            same &= row[3] == f"{mine.layer_3_pooled_pct:.2f} ({mine.layer_3_sd_across_c50_pct_points:.2f})"
        check.add(same, f"{other.name} (made from the summary CSVs by generate_m1_reference_figures.py): the same values at two decimals")
    # sparse rule and colors
    for layer in LAYERS:
        crossing, _, distinct = res.cells(layer)
        check.lines.append(f"info {layer}: non-empty cells {int((crossing > 0).sum())}, empty {int((crossing == 0).sum())}, marked sparse {int(((distinct > 0) & (distinct < SPARSE_BELOW)).sum())}" + (f"; counting pooled observations instead of distinct pairs, {int(((crossing > 0) & (crossing < SPARSE_BELOW)).sum())} would be sparse" if layer == "Layer 3" else ""))
        check.lines.append(f"info {layer}: Lorenz curves intersect for {res.crossing_rate(layer):.2f}% of pairs" + (" (pooled observations)" if layer == "Layer 3" else ""))
    check.lines.append(f"info color scale 0 to {top:g}%: {above} panel cells above it, all sparse, are drawn at the top color")
    check.lines.append(f"info computing step: {float(res.data['compute_seconds']):.0f} s")
    return check


# ------------------------------------------------------------------ provenance
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest():
    inputs = [("layer 1 sample", L1_PATH), ("layer 2 sample", L2_PATH), ("layer 3 policies and twin outputs", L3_X_PATH)]
    inputs += [("layer 3 outcome shares", path) for path in sorted(L3_DIR.glob("*_y.parquet"))]
    inputs += [("project code", ROOT / name) for name in ("impact_function.py", "lorenz.py", "metrics.py", "sampling.py", "utils.py", "visualize.py")]
    if TWIN_STEP.exists():
        inputs.append(("twin code that wrote the layer 3 shares (private repository)", TWIN_STEP))
    rows = []
    for role, path in inputs:
        stat = path.stat()
        rows.append({"role": role, "file": path.name, "bytes": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"), "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(HERE / "input_manifest.csv", index=False)
    return pd.DataFrame(rows)


def write_captions(top, above, surface_top):
    note = rf" Colors above {top:g}\% occur only in sparse cells and are drawn at the top color." if above else ""
    n = len(PHI_GRID)
    text = rf"""% Figure captions. Generated by m1_reference.py. Requires graphicx.
% Reference heatmap: figures/m1_reference_heatmap.pdf
\caption{{Pairwise metric disagreement over the Lorenz-geometry grid, at an illustrative stakeholder reference profile that is not a universal or empirically elicited preference. Rows show Gini vs Theil, and Gini and Theil vs Atkinson at $\varepsilon=1$. Layer~1 has no $\varphi$ parameters; Layers~2 and~3 use $\varphi_{{\mathrm{{mode}}}}=\varphi_{{\mathrm{{time}}}}=0.5$. Each cell gives the share of disagreeing pairs by the Zheng descriptor $Z$ and the Lorenz-intersection measure $\Gamma$, in cells of width 0.05. Lorenz-dominance pairs are excluded from the within-cell rate. Panel headings give the rate over all sampled pairs, including dominance pairs. All panels share one color scale. Dots mark cells with fewer than {SPARSE_BELOW} non-dominance pairs, and blank cells hold none. Layer~3 pools 100 $c_{{50}}$ realizations of 25,000 policy pairs, and its outcome shares are relative to whole-day baseline trips; a sparse cell there has fewer than {SPARSE_BELOW} distinct policy pairs.{note}}}

% Disagreement against epsilon: figures/m1_reference_pairwise_vs_epsilon.pdf
\caption{{Pairwise metric disagreement against the Atkinson parameter $\varepsilon$, as a percentage of all sampled pairs. Layer~1 has no $\varphi$ parameters; Layers~2 and~3 fix $\varphi_{{\mathrm{{mode}}}}=\varphi_{{\mathrm{{time}}}}=0.5$. The Gini vs Theil rate does not depend on $\varepsilon$ and is a constant reference line. The vertical line marks the illustrative reference $\varepsilon=1$. In Layer~3, bands and bars show $\pm1$ standard deviation across 100 $c_{{50}}$ realizations; they describe variation across realizations and are not confidence intervals.}}

% Disagreement against phi_mode: figures/m1_reference_pairwise_vs_phi_mode.pdf
\caption{{Pairwise metric disagreement against $\varphi_{{\mathrm{{mode}}}}$, with $\varepsilon=1$ and $\varphi_{{\mathrm{{time}}}}=0.5$, as a percentage of all sampled pairs. The vertical line marks the illustrative reference $\varphi_{{\mathrm{{mode}}}}=0.5$. Lines join {n} evaluated values, in steps of {PHI_STEP:g}. In Layer~3, bands and bars show $\pm1$ standard deviation across 100 $c_{{50}}$ realizations; they describe variation across realizations and are not confidence intervals.}}

% Disagreement against phi_time: figures/m1_reference_pairwise_vs_phi_time.pdf
\caption{{Pairwise metric disagreement against $\varphi_{{\mathrm{{time}}}}$, with $\varepsilon=1$ and $\varphi_{{\mathrm{{mode}}}}=0.5$, as a percentage of all sampled pairs. The vertical line marks the illustrative reference $\varphi_{{\mathrm{{time}}}}=0.5$. Lines join {n} evaluated values, in steps of {PHI_STEP:g}. In Layer~3, bands and bars show $\pm1$ standard deviation across 100 $c_{{50}}$ realizations; they describe variation across realizations and are not confidence intervals.}}

% Disagreement over the phi grid: figures/m1_reference_phi_surface.pdf
\caption{{Pairwise metric disagreement over the cross product of $\varphi_{{\mathrm{{mode}}}}$ and $\varphi_{{\mathrm{{time}}}}$, as a percentage of all sampled pairs, with Atkinson $\varepsilon=1$ ({n} values per parameter, {len(SETTINGS)} settings). Layer~1 has no $\varphi$ parameters and is not shown. All panels share one color scale, from 0 to {surface_top:g}\%. The dot marks the illustrative reference $\varphi_{{\mathrm{{mode}}}}=\varphi_{{\mathrm{{time}}}}=0.5$. Layer~3 values are pooled over 100 $c_{{50}}$ realizations.}}
"""
    (HERE / "figure_captions.tex").write_text(text, encoding="utf-8")


def write_readme(res, check, top, above, surface_top):
    denominator = json.loads((HERE / "denominator_check.json").read_text(encoding="utf-8"))
    layer3 = json.loads(pq.read_schema(sorted(L3_DIR.glob("*_y.parquet"))[0]).metadata[b"sampling"])
    c50_info = json.loads(pq.read_schema(L3_X_PATH).metadata[b"c50_sampling"])
    import matplotlib
    import pyarrow
    import scipy
    table = pd.read_csv(HERE / "m1_reference_table.csv")
    rows = "\n".join(f"| {r.comparison}{', ε = 1' if r.comparison != 'Gini vs Theil' else ''} | {r.layer_1_pct:.2f} | {r.layer_2_pct:.2f} | {r.layer_3_pooled_pct:.2f} ({r.layer_3_sd_across_c50_pct_points:.2f}) |" for r in table.itertuples())
    passed = sum(line.startswith("ok") for line in check.lines)
    text = f"""# M1 reference table and figures: provenance

Generated by `m1_reference.py` on {datetime.now().isoformat(timespec="seconds")}. To regenerate from the samples: `python results/m1_reference/m1_reference.py --recompute` (the computing step took {float(res.data['compute_seconds']) / 60:.0f} minutes).

Nothing here is read from the summary CSVs in `results/m1/`. They are only compared with the results, in `m1_reference_checks.txt`. No input file was changed.

## Reference profile

Atkinson ε = 1. In Layers 2 and 3, φ_mode = φ_time = 0.5; Layer 1 has no φ parameters. This is an illustrative stakeholder profile for visualization, not a universal or empirically elicited preference.

| Pairs disagreeing, % of all sampled pairs | Layer 1 | Layer 2 | Layer 3 pooled (SD across c50) |
|---|---:|---:|---:|
{rows}

## φ grid

Each φ takes {len(PHI_GRID)} values from 0 to 1 in steps of {PHI_STEP:g}, and Layers 2 and 3 are evaluated on their cross product: {len(SETTINGS)} (φ_mode, φ_time) settings. On this grid the three comparisons use ε = 1. The full ε grid is computed only at φ_mode = φ_time = 0.5. `results/m1/` uses φ = {", ".join(f"{v:g}" for v in OLD_GRID)}. The two grids share φ = {", ".join(f"{v:g}" for v in SHARED)}, so {len(SHARED) ** 2} settings can be compared, and their counts are identical (see Checks); the other old values are not on this grid.

## Input files

Hashes, sizes and dates of all inputs are in `input_manifest.csv`.

| Layer | File | Sample |
|---|---|---|
| 1 | `results/seed42_K52_n250000_b.parquet` | {res.n['Layer 1']:,} pairs of burden vectors, seed 42 |
| 2 | `results/seed42_K52_n250000_y.parquet` | {res.n['Layer 2']:,} pairs of synthetic outcome shares, seed 42 |
| 3 | `results/layer3/seed42_n25000_y/*_y.parquet` (100 files), from `results/layer3/seed42_n25000_x.parquet` | {int(res.data['L3_n'][0]):,} policy pairs, seed 42, at {len(res.c50)} c50 values |

All layers use the same K = 52 groups and the same population weights p. c50 values are Monte Carlo draws, uniform on [{c50_info['distribution']['range'][0]:g}, {c50_info['distribution']['range'][1]:g}), seed {c50_info['seed']}; the drawn values run from {res.c50.min():.4f} to {res.c50.max():.4f}. The twin outputs come from commit `{layer3['twin']['commit'][:7]}` of branch `{layer3['twin']['branch']}`, with uncommitted changes recorded in the metadata.

## Layer 3 denominator

Whole-day baseline trips: outcome share = trips of the group with that outcome inside the policy's window / the group's baseline trips over all {denominator['slots_in_whole_day_baseline']} daily slots.

- **Code:** `fairness/burden_step.py` in the twin repository: `baseline_zones = inflow.sum(1)` and `baseline_groups = baseline_zones @ split` use the whole day; only the numerator is restricted to the window.
- **Metadata:** every y file describes the shares as "{layer3['columns'].split('<outcome>_<group>: ')[1].split(';')[0]}", and all 100 files hold one baseline vector, totalling {res.text['L3']['baseline_total']:,.1f} trips.
- **Numeric check:** `check_layer3_denominator.py` recomputed {denominator['policies_checked']} policies at 3 c50 values with the twin's inflow. The largest gap to the stored shares is {denominator['largest_abs_gap_stored_vs_whole_day_denominator']:.1e} with the whole-day denominator and {denominator['largest_abs_gap_stored_vs_window_denominator']:.3f} with a window denominator. Results are in `denominator_check.json`.

So no M1 result needed recomputing because of the denominator. The docstring of `impact_function.py` still says "baseline trips in the time window W"; this does not apply to Layer 3, and the docstring was not changed.

## Metrics and disagreement

- **Burdens:** `impact_function.linear_impact_function`: b_k = c_k + φ_mode m_k + φ_time (a_k + p_k). Layer 1 samples b directly.
- **Metrics:** Gini, Theil T and Atkinson with the functions behind `metrics.compute_metrics`. Atkinson is the project's `bad_atkinson` for burdens (Sheriff and Maguire 2020) with β = -ε, not the income-convention `atkinson()`.
- **Normalization:** each metric is divided by its maximum for p: 1 - p_min, ln(1 / p_min), and p_min^(β / (1 - β)) - 1.
- **Disagreement:** `visualize.pairwise_disagreement`: two metrics disagree on a pair when their normalized differences M(A) - M(B) have opposite signs and neither is a tie. A tie is an absolute difference of at most {DEFAULT_TAU:g}.
- **Rates:** disagreements / all sampled pairs. Ties and Lorenz-dominance pairs stay in the denominator of the table and the line figures.

## Heatmap

- **Descriptors:** `lorenz.lorenz_measures`, curve tolerance {LORENZ_TOL:g}: Z = min(A+, A-) / (A+ + A-) and Γ = min(length of L_A < L_B, length of L_B < L_A). Dominance is Z = 0, as in `visualize.py`; this also covers coincident curves.
- **Grid:** the M1 grid of 10 × 10 cells of width 0.05 on [0, 0.5]². Cells include their lower edge, and the last one also includes 0.5. `visualize.py` defines no binning, so this comes from `results/m1/`.
- **Within-cell rate:** disagreements / non-dominance pairs in the cell. In Layer 3, each c50 realization places a pair in the cell of its own (Z, Γ), and counts are pooled.
- **Sparse cells:** fewer than {SPARSE_BELOW} non-dominance pairs, marked with a dot. In Layer 3 this counts distinct policy pairs, because pooled counts repeat the same pairs across c50 values. Rates are not affected, and both counts are in `m1_reference_heatmap_cells.csv`.
- **Color scale:** one scale from 0 to {top:g}%, the highest non-sparse rate rounded up to a multiple of 5. {f"{above} sparse panel cells above it are drawn at the top color." if above else "No cell is above it."}

The surface figure of the φ grid uses its own common scale, from 0 to {surface_top:g}%.

## Layer 3 SD

For each setting and comparison, the rate is computed separately for each of the {len(res.c50)} c50 files. The SD is the sample SD (ddof = 1) of those rates, in percentage points. It describes variation across c50 realizations and is not a confidence interval. The pooled rate is total disagreements / total observations, which equals the mean of those rates. At φ = (0, 0) the SD is 0, because only the window then affects Layer 3 verdicts.

## Checks

{passed} checks passed and {check.failed} found a discrepancy; details are in `m1_reference_checks.txt`. They include the table against the points read back from the figures, the heatmap cells against the table, the φ grid against the full computation at the reference, recomputed metrics against the columns stored in the Layer 1 and 2 files, and this recomputation against the existing summary CSVs on the settings they contain.

## Data limitations

1. **Layer 3 has fewer pairs.** {int(res.data['L3_n'][0]):,} policy pairs against {res.n['Layer 1']:,} in Layers 1 and 2. Its {res.n['Layer 3']:,} pooled observations repeat those pairs across c50 values, so they are not independent pairs.
2. **The SD covers c50 only.** Other twin parameters stay at their reference values, so it is not total epistemic uncertainty, and it omits sampling error from the policy pairs.
3. **Layer 3 coverage.** The groups cover {100 * layer3['share of AV-bound trips assigned']:.1f}% of trips entering the cordon. The rest start outside the municipality and belong to no group.
4. **Layer 2 is synthetic.** Its outcomes are independent draws, not estimates for Area Verde.
5. **Layers are not paired.** Each layer has its own sample, so comparisons are between distributions.
6. **φ grid.** Lines between grid values are visual only, and Layer 3 rates can change quickly close to φ_mode = 0, faster than a step of {PHI_STEP:g} fully resolves. The cross product uses ε = 1 only.
7. **Reference ε.** ε = 1 is the lowest Gini vs Atkinson rate on the ε grid in Layers 1 and 2, so the table alone understates how far those metrics can diverge. The ε figure shows the full range.

## Files

- `m1_reference_table.csv`, `m1_reference_table.tex`: the reference table.
- `figures/`: `m1_reference_heatmap`, `m1_reference_pairwise_vs_epsilon`, `m1_reference_pairwise_vs_phi_mode`, `m1_reference_pairwise_vs_phi_time`, `m1_reference_phi_surface`, each as PDF and PNG.
- `figure_captions.tex`: captions.
- `m1_reference_rates_long.csv`: every computed rate, with counts and ties. `m1_reference_layer3_per_c50.csv`: the Layer 3 rate at each c50 value and φ setting.
- `m1_reference_heatmap_cells.csv`: every cell.
- `m1_reference_counts.npz`: raw counts from the computing step.
- `m1_reference.py`, `check_layer3_denominator.py`: the scripts.
- `input_manifest.csv`, `denominator_check.json`, `m1_reference_checks.txt`: provenance and checks.

## Software

Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, pyarrow {pyarrow.__version__}, matplotlib {matplotlib.__version__}. The denominator check ran in the twin's WSL environment.
"""
    (HERE / "README.md").write_text(text, encoding="utf-8")


def outputs():
    setup_matplotlib()
    res = Results()
    write_table(res)
    cells = write_cells(res)
    long_rates(res).to_csv(HERE / "m1_reference_rates_long.csv", index=False, float_format="%.6f")
    write_per_c50(res)
    figures = {"epsilon": figure_epsilon(res), "phi_mode": figure_phi(res, "phi_mode"), "phi_time": figure_phi(res, "phi_time")}
    figures["surface"], surface_top = figure_surface(res)
    figures["heatmap"], top, above = figure_heatmap(res)
    check = run_checks(res, cells, figures, top, above)
    (HERE / "m1_reference_checks.txt").write_text("\n".join(check.lines) + "\n", encoding="utf-8")
    write_captions(top, above, surface_top)
    if not (HERE / "input_manifest.csv").exists():
        write_manifest()
    write_readme(res, check, top, above, surface_top)
    print(f"\n{check.failed} discrepancies. Outputs in {HERE}")


def main():
    parser = argparse.ArgumentParser(description="M1 reference table and figures, from the samples.")
    parser.add_argument("--recompute", action="store_true", help="compute the counts again from the samples")
    parser.add_argument("--workers", type=int, default=4, help="processes for the computing step")
    args = parser.parse_args()
    if args.recompute or not COUNTS.exists():
        compute(args.workers)
        write_manifest()
    outputs()


if __name__ == "__main__":
    main()
