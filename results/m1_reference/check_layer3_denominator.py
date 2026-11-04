"""Check which denominator the stored Layer 3 outcome shares actually use. Read-only; writes denominator_check.json next to this file.

For a sample of policies and three c50 values, the shares are recomputed with the twin's baseline inflow and the project's own
fairness/burden_step.py functions, once with the whole-day baseline (all 288 slots) and once with the baseline inside each
policy's window, and both are compared with the values stored in the y files. Only the denominator differs between the two.

Needs the digital twin, which is not part of this repository. Run from the twin's simulation folder, in its environment:
    python -W ignore <path to this file>
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.getcwd())
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "layer3"

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from fairness import burden_step as bs  # noqa: E402

N_POLICIES = 300
SEED = 0

x_path = RESULTS / "seed42_n25000_x.parquet"
x_file = pq.ParquetFile(x_path)
metadata = x_file.schema_arrow.metadata
x_settings = json.loads(metadata[b"sampling"])
c50_values = json.loads(metadata[b"c50_sampling"])["values"]
twin = json.loads(metadata[b"twin_evaluation"])

# the same set-up as burden_step(): groups, weights and whole-day baseline trips
inflow_by_zone, _ = bs.inflow_mod.build_out_zone_inflow("platoon")
zones = sorted(int(z) for z, series in inflow_by_zone.items() if series.sum() > 0)
m_by_zone = dict(zip(twin["zones"], twin["m_z"]))
inflow = np.stack([inflow_by_zone[z] for z in zones])
baseline_zones = inflow.sum(1)
areas, _, _, split = bs._groups(zones, baseline_zones)
baseline_groups = baseline_zones @ split
K = len(areas)
bs._STATE.update(inflow=inflow, m_z=np.array([m_by_zone[z] for z in zones]), exempted=twin["fraction_exempted"], split=split,
                 baseline_groups=baseline_groups)

which = [0, len(c50_values) // 2, len(c50_values) - 1]
y_paths = [RESULTS / "seed42_n25000_y" / f"seed{x_settings['seed']}_K{K}_n{x_settings['n_samples']}_c50-{c50_values[j]:.4f}_y.parquet" for j in which]
stored_baseline = np.array(json.loads(pq.read_schema(y_paths[0]).metadata[b"sampling"])["baseline_trips"])

block = x_file.read_row_group(0, columns=["id", *bs.LEVERS, *[f"P_r (c50 = {c50_values[j]})" for j in which],
                                          *[f"P_a (t = {t})" for t in range(bs.N_SLOTS)], *[f"P_p (t = {t})" for t in range(bs.N_SLOTS)]])
rows = np.sort(np.random.default_rng(SEED).choice(block.num_rows, N_POLICIES, replace=False))
x = np.column_stack([block[name].to_numpy() for name in bs.LEVERS])
P_r = np.column_stack([block[f"P_r (c50 = {c50_values[j]})"].to_numpy() for j in which])
P_a = np.column_stack([block[f"P_a (t = {t})"].to_numpy() for t in range(bs.N_SLOTS)])
P_p = np.column_stack([block[f"P_p (t = {t})"].to_numpy() for t in range(bs.N_SLOTS)])
names = [f"{outcome}_{k}" for outcome in bs.OUTCOMES for k in range(1, K + 1)]
stored = []
for path in y_paths:
    table = pq.ParquetFile(path).read_row_group(0, columns=["id", "w_start", *names])
    assert np.array_equal(table["id"].to_numpy(), block["id"].to_numpy())
    assert np.array_equal(table["w_start"].to_numpy(), block["w_start"].to_numpy())
    stored.append(np.column_stack([table[name].to_numpy() for name in names]))

gap_whole_day, gap_window, ratios, window_share = 0.0, 0.0, [], []
for row in rows:
    w_start, w_end = x[row, 0], x[row, 1]
    in_window = (bs.SLOT_SECONDS >= w_start * 3600.0) & (bs.SLOT_SECONDS <= w_end * 3600.0)
    counts = bs._zone_counts(w_start, w_end, P_r[row], P_a[row], P_p[row]) @ split   # (outcomes, c50 values, groups)
    window_groups = inflow[:, in_window].sum(1) @ split
    usable = window_groups > 0
    window_share.append(window_groups[usable] / baseline_groups[usable])
    for i in range(len(which)):
        values = stored[i][row].reshape(len(bs.OUTCOMES), K)
        whole_day = counts[:, i] / baseline_groups
        window = counts[:, i, usable] / window_groups[usable]
        gap_whole_day = max(gap_whole_day, float(np.abs(values - whole_day).max()))
        gap_window = max(gap_window, float(np.abs(values[:, usable] - window).max()))
        positive = values[:, usable] > 1e-12
        ratios.append(np.median((window / np.where(positive, values[:, usable], np.nan))[positive]))

share = np.concatenate(window_share)
result = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "policies_checked": N_POLICIES, "c50_values_checked": [c50_values[j] for j in which], "seed_of_the_policy_sample": SEED,
    "largest_abs_gap_stored_vs_whole_day_denominator": gap_whole_day,
    "largest_abs_gap_stored_vs_window_denominator": gap_window,
    "median_ratio_window_share_to_stored_share": float(np.median(ratios)),
    "window_baseline_as_share_of_whole_day": {"min": float(share.min()), "median": float(np.median(share)), "max": float(share.max())},
    "largest_abs_gap_baseline_trips_metadata_vs_recomputed": float(np.abs(stored_baseline - baseline_groups).max()),
    "baseline_trips_total": float(baseline_groups.sum()), "slots_in_whole_day_baseline": bs.N_SLOTS,
    "conclusion": "whole-day" if gap_whole_day < 1e-9 < gap_window else "CHECK FAILED",
}
(HERE / "denominator_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
