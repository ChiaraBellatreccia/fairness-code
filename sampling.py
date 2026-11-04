"""Sampling of the policy pairs for the three layers.

File format: two consecutive rows per pair, one per policy. Column 'id' is the pair number, the same in
both rows (0, 0, 1, 1, 2, 2, ...); the first row of a pair is policy A, the second policy B. The other
columns hold that policy's values.

Layer 1: burden pairs sampled with Latin hypercube sampling (LHS) and saved as
seed{seed}_K{K}_n{n_samples}_b.parquet, columns id, b1, ..., bK (2 * n_samples rows).

Layer 2: outcome pairs y sampled with LHS and saved as seed{seed}_K{K}_n{n_samples}_y.parquet, columns
id, cancelled_1, ..., cancelled_K, mode_shifted_1, ..., mode_shifted_K, anticipated_1, ..., anticipated_K,
postponed_1, ..., postponed_K: for each group, the shares of its baseline trips that are cancelled,
shifted to public transport, anticipated and postponed (the rest are unchanged). impact_function.py turns
them into burdens b1, ..., bK in the same file.

Layer 3: policy pairs x sampled with LHS and saved as seed{seed}_n{n_samples}_x.parquet, columns id, w_start,
w_end (window start and end, hours after midnight), fee_base (fee of Euro 0 vehicles, euros) and fee_step
(fee reduction per Euro class, euros): the fee of Euro class e = 0, ..., 6 is max(0, fee_base - e * fee_step).
The digital twin turns each policy into outcomes y.

Layer 3, epistemic parameter: c50 values sampled with Monte Carlo and added to an x file, as one empty column
'P_r (c50 = value)' per value. Each cell will hold the twin's P_r for that policy and that c50, the only one of the
twin's behaviour quantities that changes with c50.

Terminal: the first argument chooses what to sample, followed by its options.
    python sampling.py b --K 5 --n-samples 250000 --seed 42
    python sampling.py y --K 5 --n-samples 250000 --seed 42
    python sampling.py x --n-samples 250000 --seed 42
    python sampling.py c50 results/seed42_n250000_x.parquet --n-samples 100 --seed 42
    python sampling.py --help      (lists the subcommands)
    python sampling.py b --help    (lists the options of b)
Notebook:  from sampling import b_sampling; path = b_sampling(K=5, n_samples=1000)
           from sampling import y_sampling; path = y_sampling(K=5, n_samples=1000)
           from sampling import x_sampling; path = x_sampling(n_samples=1000)
           from sampling import c50_sampling; c50_sampling(path, n_samples=100)

Every run shows a progress bar while writing and ends with a summary of time, memory and disk usage
(verbose=False or --quiet turns both off).
"""

import argparse
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from numpy.typing import ArrayLike
from scipy.stats import qmc
from tqdm.auto import tqdm

from utils import RunMonitor, fmt_bytes, integer, population_weights, print_rows, usage_rows

DEFAULT_K = 5
DEFAULT_N_SAMPLES = 250_000
DEFAULT_B_INTERVAL = (0.0, 1.0)  # burdens are sampled on the left-open interval (lo, hi]
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 50_000  # pairs per parquet row group; also the step of the progress bar
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
# layer-2 outcomes of each group, as shares of its baseline trips; the unchanged share is 1 minus their sum
OUTCOMES = ("cancelled", "mode_shifted", "anticipated", "postponed")
# layer-3 policy levers: name -> (default range, unit)
POLICY_LEVERS = {
    "w_start": ((6.0, 12.0), "hours after midnight"),  # window start
    "w_end": ((14.0, 20.0), "hours after midnight"),   # window end
    "fee_base": ((0.0, 10.0), "euros"),                # fee of Euro 0 vehicles
    "fee_step": ((0.0, 3.0), "euros"),                 # fee reduction per Euro class
}
# layer-3 epistemic parameter: c50, the fee (euros) at which half of the vehicles stay rigid
DEFAULT_N_C50 = 100
DEFAULT_C50_RANGE = (4.0, 11.0)  # uniform, as in the twin (i_b_p50_cost: stats.uniform with loc 4, scale 7)


# ------------------------------------------------------------------ helpers
def _check_counts(n_samples: int, chunk_size: int, seed: int) -> tuple[int, int, int]:
    # integers only: a float such as 2.5e5 or a seed of None would give a wrong or irreproducible file name
    n_samples, chunk_size, seed = (integer(value, name) for value, name in
                                   ((n_samples, "n_samples"), (chunk_size, "chunk_size"), (seed, "seed")))
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return n_samples, chunk_size, seed


def _check_settings(K, n_samples: int, chunk_size: int, seed: int) -> tuple[int, int, int, int]:
    K = integer(K, "K")
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}")
    return (K, *_check_counts(n_samples, chunk_size, seed))


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def _uniform_left_open(u: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    """Map u in [0, 1) uniformly onto (lo, hi], in place; the LHS stratification is kept."""
    lo, hi = interval
    u *= -(hi - lo)
    u += hi  # = hi - u * (hi - lo), without a second array in memory
    return u


def _write_pairs(path: Path, rows: np.ndarray, names: list[str], settings: dict,
                 chunk_size: int, verbose: bool) -> None:
    """Write pairs to parquet: column 'id' (pair number, same in both rows of a pair), then one column per name.

    rows has shape (2 * n_pairs, len(names)); rows 2i and 2i + 1 are policies A and B of pair i,
    and both get id i. The file is written in chunks of chunk_size pairs, with settings in the metadata.
    """
    schema = pa.schema([("id", pa.int64())] + [(name, pa.float64()) for name in names],
                       metadata={"sampling": json.dumps(settings)})
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = rows.shape[0] // 2
    temporary = path.with_name(path.name + ".tmp")  # an interrupted run never leaves a half-written file
    try:
        # no dictionary encoding: it only adds size and time for values that (almost) never repeat
        with pq.ParquetWriter(temporary, schema, use_dictionary=False) as writer, \
                tqdm(total=n_pairs, unit="pairs", unit_scale=True, desc="Writing parquet", disable=not verbose) as bar:
            for start in range(0, n_pairs, chunk_size):
                stop = min(start + chunk_size, n_pairs)
                block = rows[2 * start:2 * stop]
                columns = ({"id": np.repeat(np.arange(start, stop, dtype=np.int64), 2)}
                           | {name: block[:, j] for j, name in enumerate(names)})
                writer.write_table(pa.table(columns, schema=schema))
                bar.update(stop - start)
        os.replace(temporary, path)  # fails if another program (e.g. a parquet viewer) keeps the file open
    finally:
        temporary.unlink(missing_ok=True)


def _print_summary(path: Path, n_pairs: int, data_bytes: int, monitor: RunMonitor) -> None:
    metadata = pq.read_metadata(path)
    file_bytes = path.stat().st_size
    disk = shutil.disk_usage(path.parent)
    ram = psutil.virtual_memory()
    print_rows([
        ("file", str(path)),
        ("rows x columns", f"{metadata.num_rows:,} ({n_pairs:,} pairs x 2) x {metadata.num_columns} "
                           f"({metadata.num_row_groups} row group{'s' if metadata.num_row_groups != 1 else ''})"),
        ("size on disk", f"{fmt_bytes(file_bytes)} (values in memory {fmt_bytes(data_bytes)}, "
                         f"compression ratio {data_bytes / file_bytes:.2f})"),
        *usage_rows(monitor, n_pairs),
        ("RAM available", f"{fmt_bytes(ram.available)} of {fmt_bytes(ram.total)}"),
        ("disk free", f"{fmt_bytes(disk.free)} of {fmt_bytes(disk.total)}"),
    ])


# ------------------------------------------------------------------ layer 1
def b_sampling(
    K: int = DEFAULT_K,
    n_samples: int = DEFAULT_N_SAMPLES,
    b_interval: tuple[float, float] = DEFAULT_B_INTERVAL,
    p: ArrayLike | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = OUTPUT_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Sample n_samples burden pairs for layer 1 and save them to parquet.

    Each pair is one LHS point in 2K dimensions: the first K coordinates are the burdens of
    policy A, the last K those of policy B. Every coordinate is uniform on the left-open
    interval (lo, hi]. The file has two consecutive rows per pair, columns id (pair number,
    the same in both rows; the first row is A), b1, ..., bK.

    The population weights p (default 1/K) do not affect the sampling. They are stored with
    the other settings in the parquet metadata under the key "sampling", so that later steps
    read the same p.

    The file is written in row groups of chunk_size pairs. With verbose=True, a progress bar
    is shown while writing and a summary of time, memory and disk usage is printed at the end.

    Returns the path of the saved file.
    """
    K, n_samples, chunk_size, seed = _check_settings(K, n_samples, chunk_size, seed)
    lo, hi = b_interval
    if not 0 <= lo < hi:
        raise ValueError(f"b_interval must satisfy 0 <= lo < hi, got {b_interval}")
    p = population_weights(p, K)
    path = Path(output_dir) / f"seed{seed}_K{K}_n{n_samples}_b.parquet"

    data_bytes = n_samples * 2 * K * np.dtype(float).itemsize
    if verbose:
        print(f"Layer 1: {n_samples:,} burden pairs, K={K}, b in ({lo}, {hi}], seed {seed}")
        print(f"  {2 * n_samples:,} rows x {K} burdens = {fmt_bytes(data_bytes)} in memory "
              f"({fmt_bytes(psutil.virtual_memory().available)} RAM available)")
        print("  Sampling with LHS (all pairs at once, as the strata depend on n)...")

    with RunMonitor() as monitor:
        u = qmc.LatinHypercube(d=2 * K, rng=seed).random(n_samples)  # plain LHS: scramble=True, no optimization
        b = _uniform_left_open(u, (lo, hi))
        monitor.stage("LHS sampling")

        rows = b.reshape(2 * n_samples, K)  # no copy: row 2i is policy A of pair i, row 2i + 1 policy B
        names = [f"b{k}" for k in range(1, K + 1)]
        settings = {
            "layer": 1,
            "K": K,
            "n_samples": n_samples,
            "format": "two consecutive rows per pair with the same id (pair number); first row policy A, second B",
            "b_interval": [lo, hi],
            "interval_type": "left-open (lo, hi]",
            "p": p.tolist(),
            "seed": seed,
            "sampler": {"engine": "scipy.stats.qmc.LatinHypercube", "d": 2 * K, "scramble": True, "optimization": None},
            "created": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
        }
        _write_pairs(path, rows, names, settings, chunk_size, verbose)
        monitor.stage("writing parquet")

        if verbose:
            _print_summary(path, n_samples, data_bytes, monitor)
    return path


# ------------------------------------------------------------------ layer 2
def y_sampling(
    K: int = DEFAULT_K,
    n_samples: int = DEFAULT_N_SAMPLES,
    p: ArrayLike | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = OUTPUT_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Sample n_samples outcome pairs for layer 2 and save them to parquet.

    For each policy and each group k, the shares of the group's baseline trips that are cancelled,
    shifted to public transport, anticipated and postponed (the rest are unchanged) are sampled
    uniformly on the feasible set {shares >= 0, sum <= 1}: five LHS coordinates (the four outcomes
    and 'unchanged') are mapped to Gamma(1, 1) by its inverse CDF, -ln(1 - u), and divided by their
    sum, which gives Dirichlet(1, 1, 1, 1, 1). Every share is in (0, 1), with marginal Beta(1, 4)
    (mean 0.2); the division keeps the LHS stratification of the five coordinates only in part.
    Each group has its own LHS in 10 dimensions (2 policies x 5 shares), seeded from seed: the same
    distribution as one LHS over all groups, with far less memory.

    The file has two consecutive rows per pair, columns id (pair number, the same in both rows; the
    first row is A), then cancelled_1..cancelled_K, mode_shifted_1..K, anticipated_1..K, postponed_1..K.
    The population weights p (default 1/K) are stored in the metadata as in b_sampling.

    With verbose=True, progress bars and a summary of time, memory and disk usage are shown.

    Returns the path of the saved file.
    """
    K, n_samples, chunk_size, seed = _check_settings(K, n_samples, chunk_size, seed)
    p = population_weights(p, K)
    path = Path(output_dir) / f"seed{seed}_K{K}_n{n_samples}_y.parquet"
    n_outcomes = len(OUTCOMES)

    data_bytes = n_samples * 2 * n_outcomes * K * np.dtype(float).itemsize
    if verbose:
        print(f"Layer 2: {n_samples:,} outcome pairs, K={K}, {n_outcomes} shares per group "
              f"(uniform on shares >= 0, sum <= 1), seed {seed}")
        print(f"  {2 * n_samples:,} rows x {n_outcomes * K} shares = {fmt_bytes(data_bytes)} in memory "
              f"({fmt_bytes(psutil.virtual_memory().available)} RAM available)")

    with RunMonitor() as monitor:
        shares = np.empty((n_samples, 2, n_outcomes, K))  # pair, policy (A, B), outcome, group
        group_seeds = np.random.SeedSequence(seed).spawn(K)
        for k in tqdm(range(K), unit="group", desc="Sampling with LHS", disable=not verbose):
            u = qmc.LatinHypercube(d=2 * (n_outcomes + 1), rng=np.random.default_rng(group_seeds[k])).random(n_samples)
            gamma = -np.log1p(-u.reshape(n_samples, 2, n_outcomes + 1))  # inverse CDF of Gamma(1, 1)
            gamma /= gamma.sum(axis=2, keepdims=True)  # Dirichlet(1, ..., 1); the last share is 'unchanged'
            shares[:, :, :, k] = gamma[:, :, :n_outcomes]
        monitor.stage("LHS sampling")

        rows = shares.reshape(2 * n_samples, n_outcomes * K)  # no copy: rows alternate A, B; columns outcome by outcome
        names = [f"{outcome}_{k}" for outcome in OUTCOMES for k in range(1, K + 1)]
        settings = {
            "layer": 2,
            "K": K,
            "n_samples": n_samples,
            "format": "two consecutive rows per pair with the same id (pair number); first row policy A, second B",
            "outcomes": list(OUTCOMES),
            "columns": "<outcome>_<group>, shares of the group's baseline trips; the rest are unchanged",
            "distribution": "uniform on {shares >= 0, sum <= 1}: Dirichlet(1, 1, 1, 1, 1) over the four outcomes "
                            "and 'unchanged', from LHS through the Gamma(1, 1) inverse CDF, normalized",
            "p": p.tolist(),
            "seed": seed,
            "sampler": {"engine": "scipy.stats.qmc.LatinHypercube", "d": 2 * (n_outcomes + 1), "per": "group",
                        "group seeds": "numpy.random.SeedSequence(seed).spawn(K)", "scramble": True,
                        "optimization": None},
            "created": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
        }
        _write_pairs(path, rows, names, settings, chunk_size, verbose)
        monitor.stage("writing parquet")

        if verbose:
            _print_summary(path, n_samples, data_bytes, monitor)
    return path


# ------------------------------------------------------------------ layer 3
def x_sampling(
    n_samples: int = DEFAULT_N_SAMPLES,
    w_start: tuple[float, float] = POLICY_LEVERS["w_start"][0],
    w_end: tuple[float, float] = POLICY_LEVERS["w_end"][0],
    fee_base: tuple[float, float] = POLICY_LEVERS["fee_base"][0],
    fee_step: tuple[float, float] = POLICY_LEVERS["fee_step"][0],
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = OUTPUT_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Sample n_samples policy pairs for layer 3 and save them to parquet.

    A policy has four levers: window start and end (w_start, w_end, hours after midnight), the fee
    of Euro 0 vehicles (fee_base, euros) and the fee reduction per Euro class (fee_step, euros), so
    that the fee of Euro class e = 0, ..., 6 is max(0, fee_base - e * fee_step). Each pair is one
    LHS point in 8 dimensions: the four levers of policy A, then those of policy B. Every lever is
    uniform on the left-open interval (lo, hi] of its range.

    The start range must end before the end range begins, so every sampled window has
    w_end > w_start (with the default ranges, windows last from 2 to 14 hours).

    The file has two consecutive rows per pair, columns id (pair number, the same in both rows;
    the first row is A), w_start, w_end, fee_base, fee_step. With verbose=True, a progress bar
    and a summary of time, memory and disk usage are shown.

    Returns the path of the saved file.
    """
    n_samples, chunk_size, seed = _check_counts(n_samples, chunk_size, seed)
    ranges = {"w_start": w_start, "w_end": w_end, "fee_base": fee_base, "fee_step": fee_step}
    for name, (lo, hi) in ranges.items():
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"the range of {name} must satisfy lo < hi, got {(lo, hi)}")
    if w_start[0] < 0 or w_end[1] > 24:
        raise ValueError(f"window times must lie within 0-24 hours, got w_start {w_start} and w_end {w_end}")
    if fee_base[0] < 0 or fee_step[0] < 0:
        raise ValueError(f"fees must be non-negative, got fee_base {fee_base} and fee_step {fee_step}")
    if w_start[1] >= w_end[0]:
        raise ValueError(f"the window-start range {w_start} must end before the window-end range {w_end} begins, "
                         f"otherwise some sampled windows would have w_end <= w_start")
    path = Path(output_dir) / f"seed{seed}_n{n_samples}_x.parquet"
    n_levers = len(ranges)

    data_bytes = n_samples * 2 * n_levers * np.dtype(float).itemsize
    if verbose:
        print(f"Layer 3: {n_samples:,} policy pairs, seed {seed}: "
              + ", ".join(f"{name} in ({lo:g}, {hi:g}]" for name, (lo, hi) in ranges.items())
              + " (hours / EUR)")

    with RunMonitor() as monitor:
        u = qmc.LatinHypercube(d=2 * n_levers, rng=seed).random(n_samples)  # plain LHS: scramble=True, no optimization
        levers = u.reshape(n_samples, 2, n_levers)  # pair, policy (A, B), lever
        for j, interval in enumerate(ranges.values()):
            _uniform_left_open(levers[:, :, j], interval)  # in place, on a view of u
        monitor.stage("LHS sampling")

        rows = u.reshape(2 * n_samples, n_levers)  # no copy: rows alternate A, B
        settings = {
            "layer": 3,
            "n_samples": n_samples,
            "format": "two consecutive rows per pair with the same id (pair number); first row policy A, second B",
            "levers": {name: {"range": list(ranges[name]), "unit": POLICY_LEVERS[name][1]} for name in ranges},
            "interval_type": "left-open (lo, hi]",
            "fees": "fee of Euro class e = 0, ..., 6: max(0, fee_base - e * fee_step)",
            "seed": seed,
            "sampler": {"engine": "scipy.stats.qmc.LatinHypercube", "d": 2 * n_levers, "scramble": True,
                        "optimization": None},
            "created": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
        }
        _write_pairs(path, rows, list(ranges), settings, chunk_size, verbose)
        monitor.stage("writing parquet")

        if verbose:
            _print_summary(path, n_samples, data_bytes, monitor)
    return path


def c50_sampling(
    x_path: str | Path,
    n_samples: int = DEFAULT_N_C50,
    c50_range: tuple[float, float] = DEFAULT_C50_RANGE,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Sample n_samples values of c50 with Monte Carlo and add them to an x file written by x_sampling.

    c50 is uniform on c50_range, the distribution of the digital twin. The same values apply to every
    policy. The file gets one column 'P_r (c50 = value)' per value, in the order of the draws, with an
    empty (null) cell in every row: each cell will later hold the twin's P_r for that policy and that c50.
    The existing columns are kept. The values and the settings are stored in the metadata under the key
    "c50_sampling".

    The file is rewritten through a temporary file, so it is never left half-written. A file that already
    has P_r columns is refused, because sampling again would discard them. With verbose=True, a progress
    bar and a summary of time, memory and disk usage are shown.

    Returns the path of the updated file.
    """
    path = Path(x_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    n_samples, chunk_size, seed = _check_counts(n_samples, chunk_size, seed)
    lo, hi = c50_range
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError(f"c50_range must satisfy 0 < lo < hi, got {c50_range}")

    with RunMonitor() as monitor:
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        if "levers" not in json.loads(metadata.get(b"sampling", b"{}")):
            raise ValueError(f"{path.name} is not an x file (write one with  python sampling.py x)")
        existing = [name for name in table.column_names if name.startswith("P_r (")]
        if existing:
            raise ValueError(f"{path.name} already has {len(existing)} P_r columns; sampling again would discard them")
        n_rows, data_bytes = table.num_rows, table.nbytes
        monitor.stage("reading parquet")

        if verbose:
            print(f"Layer 3: {n_samples:,} c50 values with Monte Carlo, uniform on [{lo:g}, {hi:g}), seed {seed}; "
                  f"{n_samples:,} empty P_r columns for {n_rows:,} policies")
        # a child of the seed: x_sampling draws from the seed itself, and the two samples must be independent
        values = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]).uniform(lo, hi, n_samples).tolist()
        monitor.stage("Monte Carlo sampling")

        for value in values:
            table = table.append_column(f"P_r (c50 = {value})", pa.nulls(n_rows, pa.float64()))
        metadata[b"c50_sampling"] = json.dumps({
            "n_samples": n_samples,
            "distribution": {"name": "uniform", "range": [lo, hi], "source": "digital twin, i_b_p50_cost"},
            "sampler": "Monte Carlo, numpy.random.Generator.uniform",
            "seed": seed,
            "values": values,
            "columns": "'P_r (c50 = value)': empty, to be filled with the twin's P_r for that policy and that c50",
            "created": datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
        })
        table = table.replace_schema_metadata(metadata)
        n_pairs = n_rows // 2
        temporary = path.with_name(path.name + ".tmp")  # an interrupted run never leaves a half-written file
        try:
            with pq.ParquetWriter(temporary, table.schema, use_dictionary=False) as writer, \
                    tqdm(total=n_pairs, unit="pairs", unit_scale=True, desc="Writing parquet", disable=not verbose) as bar:
                for start in range(0, n_pairs, chunk_size):
                    writer.write_table(table.slice(2 * start, 2 * chunk_size))
                    bar.update(min(chunk_size, n_pairs - start))
            os.replace(temporary, path)  # fails if another program (e.g. a parquet viewer) keeps the file open
        finally:
            temporary.unlink(missing_ok=True)
        monitor.stage("writing parquet")

        if verbose:
            _print_summary(path, n_pairs, data_bytes, monitor)
    return path


# --------------------------------------------------------------------- main
def _run_b(args: argparse.Namespace) -> Path:
    return b_sampling(args.K, args.n_samples, tuple(args.b_interval), args.p, args.seed, args.output_dir,
                      args.chunk_size, not args.quiet)


def _run_y(args: argparse.Namespace) -> Path:
    return y_sampling(args.K, args.n_samples, args.p, args.seed, args.output_dir, args.chunk_size, not args.quiet)


def _run_x(args: argparse.Namespace) -> Path:
    return x_sampling(args.n_samples, tuple(args.w_start), tuple(args.w_end), tuple(args.fee_base),
                      tuple(args.fee_step), args.seed, args.output_dir, args.chunk_size, not args.quiet)


def _run_c50(args: argparse.Namespace) -> Path:
    return c50_sampling(args.x_path, args.n_samples, tuple(args.c50_range), args.seed, args.chunk_size, not args.quiet)


def main() -> None:
    # options shared by every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="number of pairs")
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    common.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="pairs per parquet row group and per progress-bar step")
    common.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    # options of the layers defined on population groups (b and y)
    groups = argparse.ArgumentParser(add_help=False)
    groups.add_argument("--K", type=int, default=DEFAULT_K, help="number of population groups")
    groups.add_argument("--p", type=float, nargs="+", default=None,
                        help="population weights, K values summing to 1 (default 1/K)")

    parser = argparse.ArgumentParser(description="Sample policy pairs for one layer and save them to parquet.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True, title="what to sample")

    b = subcommands.add_parser("b", parents=[common, groups], help="layer 1: burden pairs sampled with LHS",
                               description="Sample layer-1 burden pairs with LHS and save them to parquet.")
    b.add_argument("--b-interval", type=float, nargs=2, default=DEFAULT_B_INTERVAL, metavar=("LO", "HI"),
                   help="burdens are sampled on (LO, HI]")
    b.set_defaults(run=_run_b)

    y = subcommands.add_parser("y", parents=[common, groups],
                               help="layer 2: outcome pairs (four shares per group) with LHS",
                               description="Sample layer-2 outcome pairs with LHS: for each group, the shares of trips "
                                           "cancelled, mode-shifted, anticipated and postponed, uniform on "
                                           "{shares >= 0, sum <= 1}. Save them to parquet.")
    y.set_defaults(run=_run_y)

    x = subcommands.add_parser("x", parents=[common], help="layer 3: policy pairs (four levers) with LHS",
                               description="Sample layer-3 policy pairs with LHS: window start and end (hours after "
                                           "midnight), fee of Euro 0 vehicles and fee reduction per Euro class (euros). "
                                           "Save them to parquet.")
    for name, ((lo, hi), unit) in POLICY_LEVERS.items():
        x.add_argument(f"--{name.replace('_', '-')}", type=float, nargs=2, default=(lo, hi), metavar=("LO", "HI"),
                       help=f"{name} is sampled on (LO, HI], in {unit} (default {lo:g} {hi:g})")
    x.set_defaults(run=_run_x)

    c50 = subcommands.add_parser("c50", help="layer 3: c50 values with Monte Carlo, added to an x file",
                                 description="Sample c50 values with Monte Carlo, uniform as in the digital twin, and add "
                                             "one empty column 'P_r (c50 = value)' per value to an x file.")
    c50.add_argument("x_path", type=Path, help="x file written by  python sampling.py x")
    c50.add_argument("--n-samples", type=int, default=DEFAULT_N_C50, help="number of c50 values")
    c50.add_argument("--c50-range", type=float, nargs=2, default=DEFAULT_C50_RANGE, metavar=("LO", "HI"),
                     help="c50 is uniform on [LO, HI)")
    c50.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c50.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                     help="pairs per parquet row group and per progress-bar step")
    c50.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    c50.set_defaults(run=_run_c50)

    args = parser.parse_args()
    path = args.run(args)
    if args.quiet:
        print(f"Saved {args.n_samples} {'c50 values' if args.subcommand == 'c50' else 'pairs'} to {path}")


if __name__ == "__main__":
    main()
