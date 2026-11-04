"""Impact functions: from the outcomes of a policy for each group to the group burdens b.

linear_impact_function: for each group k,

    b_k = c_k + w_mode * m_k + w_time * (a_k + p_k)

    c_k  share of group k's trips that are cancelled (suppressed trips)
    m_k  share shifted to public transport (mode shift)
    a_k  share anticipated in time
    p_k  share postponed in time

All shares are relative to group k's baseline trips in the time window W, so c_k + m_k + a_k + p_k <= 1
(the rest are unchanged trips) and b_k is in [0, 1]. w_mode and w_time in [0, 1] are normative weights,
the same for all groups: how much a mode shift and a time shift count as a burden, relative to a
cancelled trip (weight 1). w_mode = w_time = 0 gives b = c, the pure suppressed trips.

In the code the shares are called cancelled, mode_shifted, anticipated and postponed: p is the population
weights everywhere else in the project.

apply_impact_function reads an outcome file written by sampling.py y, computes b1, ..., bK for every row
with linear_impact_function and adds them to the same file, so lorenz.py, metrics.py and visualize.py run
on it as on a layer-1 file. The weights are fixed for now (DEFAULT_W_MODE, DEFAULT_W_TIME).

Terminal: the first argument chooses what to run, followed by its options.
    python impact_function.py linear --cancelled 0.10 0.20 --mode-shifted 0.05 0.10 \
        --anticipated 0.02 0.00 --postponed 0.03 0.05 --w-mode 0.5 --w-time 0.25
    python impact_function.py apply results/seed42_K5_n250000_y.parquet
    python impact_function.py apply results/seed42_K5_n250000_y.parquet --w-mode 0.5 --w-time 0.25
    python impact_function.py linear --help
Notebook:
    from impact_function import linear_impact_function, apply_impact_function
    b = linear_impact_function(cancelled, mode_shifted, anticipated, postponed, w_mode=0.5, w_time=0.25)
    apply_impact_function("results/seed42_K5_n250000_y.parquet")
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from sampling import OUTCOMES  # column prefixes of the outcome files: cancelled_1, ..., postponed_K
from utils import RunMonitor, fmt_bytes, integer, print_rows, usage_rows

SHARE_SUM_TOL = 1e-9  # the four shares of a group may exceed 1 by at most this (rounding)
# fixed placeholder weights for now (a normative choice; they will be sampled later)
DEFAULT_W_MODE = 0.5
DEFAULT_W_TIME = 0.25
DEFAULT_CHUNK_SIZE = 100_000  # rows per computation step and per parquet row group
# columns computed from b by lorenz.py and metrics.py: stale once b changes
B_DEPENDENT = ("zheng", "LDI", "GINI", "Theil", "GINI_norm", "Theil_norm")


def _shares(values: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must be shares in [0, 1]")
    return values


def _weight(value: float, name: str) -> float:
    value = float(value)
    if not (np.isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def linear_impact_function(
    cancelled: ArrayLike,
    mode_shifted: ArrayLike,
    anticipated: ArrayLike,
    postponed: ArrayLike,
    *,
    w_mode: float,
    w_time: float,
) -> np.ndarray:
    """Group burdens b_k = c_k + w_mode * m_k + w_time * (a_k + p_k), in [0, 1].

    cancelled (c), mode_shifted (m), anticipated (a), postponed (p): shares of each group's baseline
    trips in the time window W; one value per group, all with the same shape. Extra leading axes
    evaluate many outcome vectors at once, e.g. shape (n, K).
    w_mode, w_time: normative weights in [0, 1], the same for all groups. They must be given
    explicitly; w_mode = w_time = 0 gives b = c (pure suppressed trips).

    Raises ValueError if a share is outside [0, 1], if c_k + m_k + a_k + p_k > 1 for some group, or if
    a weight is outside [0, 1].
    """
    shares = [_shares(values, name) for values, name in ((cancelled, "cancelled"), (mode_shifted, "mode_shifted"),
                                                         (anticipated, "anticipated"), (postponed, "postponed"))]
    if len({values.shape for values in shares}) != 1:
        raise ValueError(f"the four shares must have the same shape, got {[values.shape for values in shares]}")
    cancelled, mode_shifted, anticipated, postponed = shares
    total = cancelled + mode_shifted + anticipated + postponed
    too_large = total > 1 + SHARE_SUM_TOL
    if np.any(too_large):
        first = tuple(int(i) for i in np.argwhere(too_large)[0])
        count = np.count_nonzero(too_large)
        raise ValueError(f"cancelled + mode_shifted + anticipated + postponed must be at most 1 for every group; "
                         f"{count} {'entry exceeds' if count == 1 else 'entries exceed'} it "
                         f"(e.g. index {first}: {total[first]:.6g})")
    w_mode, w_time = _weight(w_mode, "w_mode"), _weight(w_time, "w_time")
    b = cancelled + w_mode * mode_shifted + w_time * (anticipated + postponed)
    return np.minimum(b, 1.0)  # b <= c + m + a + p <= 1; only rounding within SHARE_SUM_TOL can pass 1


def apply_impact_function(
    path: str | Path,
    w_mode: float = DEFAULT_W_MODE,
    w_time: float = DEFAULT_W_TIME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Add the burdens b1, ..., bK to an outcome file written by sampling.py y, with linear_impact_function.

    Every row (one policy) gets b_k = cancelled_k + w_mode * mode_shifted_k + w_time * (anticipated_k
    + postponed_k); the outcome columns are kept. The result has the usual format (id, b1, ..., bK and
    the sampling metadata with K and p), so lorenz.py, metrics.py and visualize.py work on the file.

    Existing b columns are replaced. Columns computed from earlier burdens (zheng, LDI, GINI, Theil,
    bad_Atk(...) and their _norm versions) are removed, because they no longer match the new b: run
    lorenz.py and metrics.py again. The weights are recorded in the metadata under "impact". The file
    is rewritten through a temporary file, so it is never left half-written. With verbose=True a
    progress bar and a run summary are shown.

    Returns the path of the updated file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    chunk_size = integer(chunk_size, "chunk_size")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    w_mode, w_time = _weight(w_mode, "w_mode"), _weight(w_time, "w_time")

    size_before = path.stat().st_size
    with RunMonitor() as monitor:
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        if b"sampling" not in metadata:
            raise ValueError(f"{path.name} has no 'sampling' metadata: it was not written by sampling.py")
        K = json.loads(metadata[b"sampling"])["K"]
        names = {outcome: [f"{outcome}_{k}" for k in range(1, K + 1)] for outcome in OUTCOMES}
        missing = [name for columns in names.values() for name in columns if name not in table.column_names]
        if missing:
            raise ValueError(f"{path.name} has no outcome columns {missing[:4]}{' ...' if len(missing) > 4 else ''}: "
                             f"it is not an outcome file (write one with  python sampling.py y)")
        outcomes = {outcome: np.column_stack([table[name].to_numpy() for name in columns])
                    for outcome, columns in names.items()}
        n_rows = table.num_rows
        monitor.stage("reading parquet")

        if verbose:
            print(f"Linear impact function on {n_rows:,} rows, K={K}: w_mode = {w_mode:g}, w_time = {w_time:g}")
        b = np.empty((n_rows, K))
        with tqdm(total=n_rows, unit="rows", unit_scale=True, desc="Computing b", disable=not verbose) as bar:
            for start in range(0, n_rows, chunk_size):
                rows = slice(start, start + chunk_size)
                b[rows] = linear_impact_function(*(outcomes[outcome][rows] for outcome in OUTCOMES),
                                                 w_mode=w_mode, w_time=w_time)
                bar.update(len(b[rows]))
        monitor.stage("computing b")

        b_names = [f"b{k}" for k in range(1, K + 1)]
        stale = [name for name in table.column_names if name in B_DEPENDENT or name.startswith("bad_Atk(")]
        table = table.drop_columns([name for name in table.column_names if name in b_names] + stale)
        for k, name in enumerate(b_names):
            table = table.append_column(name, pa.array(b[:, k]))
        for key in (b"lorenz", b"metrics"):  # describe the removed columns
            metadata.pop(key, None)
        metadata[b"impact"] = json.dumps({
            "function": "linear",
            "formula": "b_k = cancelled_k + w_mode * mode_shifted_k + w_time * (anticipated_k + postponed_k)",
            "w_mode": w_mode, "w_time": w_time,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        table = table.replace_schema_metadata(metadata)
        temporary = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, temporary, row_group_size=chunk_size, use_dictionary=False)
            os.replace(temporary, path)  # fails if another program (e.g. a parquet viewer) keeps the file open
        finally:
            temporary.unlink(missing_ok=True)
        monitor.stage("writing parquet")

        if verbose:
            print_rows([
                ("file", str(path)),
                ("rows", f"{n_rows:,} ({n_rows // 2:,} pairs)"),
                ("columns", f"b1 ... b{K} added" + (f"; removed {len(stale)} columns computed from the old b "
                                                    f"(run lorenz.py and metrics.py again)" if stale else "")),
                ("b", f"mean {b.mean():.4f}, min {b.min():.4f}, max {b.max():.4f}"),
                *usage_rows(monitor, n_rows, unit="rows"),
                ("file size", f"{fmt_bytes(size_before)} -> {fmt_bytes(path.stat().st_size)}"),
            ])
    return path


# --------------------------------------------------------------------- main
def _run_linear(args: argparse.Namespace) -> None:
    b = linear_impact_function(args.cancelled, args.mode_shifted, args.anticipated, args.postponed,
                               w_mode=args.w_mode, w_time=args.w_time)
    print(f"Linear impact function, w_mode = {args.w_mode:g}, w_time = {args.w_time:g}")
    print(f"  {'group':>5}  {'cancelled':>9}  {'mode':>6}  {'anticip.':>8}  {'postp.':>6}  {'b':>7}")
    for k, row in enumerate(zip(args.cancelled, args.mode_shifted, args.anticipated, args.postponed, b), start=1):
        print(f"  {k:>5}  {row[0]:>9.4f}  {row[1]:>6.4f}  {row[2]:>8.4f}  {row[3]:>6.4f}  {row[4]:>7.4f}")


def _run_apply(args: argparse.Namespace) -> None:
    path = apply_impact_function(args.path, args.w_mode, args.w_time, args.chunk_size, not args.quiet)
    if args.quiet:
        print(f"Added b1..bK to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Impact functions: from group outcomes to group burdens.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True, title="which impact function")

    linear = subcommands.add_parser(
        "linear", help="b_k = c_k + w_mode m_k + w_time (a_k + p_k)",
        description="Linear impact function b_k = c_k + w_mode * m_k + w_time * (a_k + p_k), one value per group.")
    for option, meaning in (("--cancelled", "share of trips cancelled (c_k)"),
                            ("--mode-shifted", "share shifted to public transport (m_k)"),
                            ("--anticipated", "share anticipated in time (a_k)"),
                            ("--postponed", "share postponed in time (p_k)")):
        linear.add_argument(option, type=float, nargs="+", required=True, metavar="SHARE",
                            help=f"{meaning}, one value per group")
    linear.add_argument("--w-mode", type=float, required=True, help="weight of a mode shift, in [0, 1]")
    linear.add_argument("--w-time", type=float, required=True, help="weight of a time shift, in [0, 1]")
    linear.set_defaults(run=_run_linear)

    apply = subcommands.add_parser(
        "apply", help="add b1..bK to an outcome file (sampling.py y) with the linear impact function",
        description="Compute b_k = cancelled_k + w_mode * mode_shifted_k + w_time * (anticipated_k + postponed_k) "
                    "for every row of an outcome file and add the columns b1..bK to it.")
    apply.add_argument("path", type=Path, help="outcome file written by  python sampling.py y")
    apply.add_argument("--w-mode", type=float, default=DEFAULT_W_MODE,
                       help=f"weight of a mode shift, in [0, 1] (default {DEFAULT_W_MODE})")
    apply.add_argument("--w-time", type=float, default=DEFAULT_W_TIME,
                       help=f"weight of a time shift, in [0, 1] (default {DEFAULT_W_TIME})")
    apply.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help="rows per computation step and per parquet row group")
    apply.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    apply.set_defaults(run=_run_apply)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
