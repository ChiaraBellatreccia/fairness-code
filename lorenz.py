"""Lorenz-curve descriptors of the policy pairs.

lorenz_classification reads a pairs file written by sampling.py (two rows per pair sharing the same 'id', the pair
number; burden columns b1, ..., bK; population weights p in the metadata) and, for every pair with Lorenz curves
L_A (first row of the pair) and L_B (second row), computes

    zheng  Z = min(A+, A-) / (A+ + A-), with A+ = area where L_A > L_B and A- = area where L_A < L_B
           (symmetric version of Zheng's 2018 almost-Lorenz-dominance ratio);
    LDI    Gamma = min(length of {L_A < L_B}, length of {L_B < L_A}) (symmetric Lorenz dominance index).

Both are 0 under Lorenz dominance in either direction and for coincident curves. The two columns 'zheng' and
'LDI' are written into the same parquet file (replaced if they are already there); the two rows of a pair get
the same values.

Terminal: the first argument chooses what to run, followed by its options.
    python lorenz.py classification results/seed42_K5_n250000_b.parquet
    python lorenz.py classification results/seed42_K5_n250000_b.parquet --test   (unit tests first)
    python lorenz.py classification --help
Notebook:
    from lorenz import lorenz_classification
    lorenz_classification("results/seed42_K5_n250000_b.parquet", test=True)
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

from utils import RunMonitor, fmt_bytes, integer, population_weights, print_rows, usage_rows

DEFAULT_TOL = 1e-12  # |L_A - L_B| below this is treated as 0, so coincident points are not read as crossings
DEFAULT_CHUNK_SIZE = 50_000  # pairs per computation step; a parquet row group holds 2 * chunk_size rows
MEASURE_COLUMNS = ("zheng", "LDI")


# ------------------------------------------------------------ Lorenz curves
def lorenz_curve(b: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knots of the population-weighted Lorenz curves of the rows of b (shape (n, K)).

    Groups are sorted from least to most burdened. Returns x (cumulative population share) and
    y (cumulative burden share), both of shape (n, K + 1), from (0, 0) to (1, 1).
    """
    order = np.argsort(b, axis=1, kind="stable")
    b_sorted = np.take_along_axis(b, order, axis=1)
    p_sorted = p[order]
    mass = p_sorted * b_sorted
    zeros = np.zeros((b.shape[0], 1))
    x = np.hstack([zeros, np.cumsum(p_sorted, axis=1)])
    y = np.hstack([zeros, np.cumsum(mass, axis=1) / mass.sum(axis=1, keepdims=True)])
    x[:, -1] = y[:, -1] = 1.0  # exact end point despite rounding in the cumulative sums
    return x, y


def _evaluate(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Values at the points t (n, m) of the piecewise-linear curves with knots x, y (n, K + 1)."""
    segment = (t[:, :, None] >= x[:, None, 1:-1]).sum(axis=2)
    x0, x1 = np.take_along_axis(x, segment, axis=1), np.take_along_axis(x, segment + 1, axis=1)
    y0, y1 = np.take_along_axis(y, segment, axis=1), np.take_along_axis(y, segment + 1, axis=1)
    return y0 + (t - x0) * (y1 - y0) / (x1 - x0)


def lorenz_measures(b_a: ArrayLike, b_b: ArrayLike, p: ArrayLike,
                    tol: float = DEFAULT_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Zheng ratio Z and Lorenz dominance index Gamma of each pair of rows of b_a and b_b.

    D = L_A - L_B is piecewise linear between the combined breakpoints of the two curves. Every
    segment where D changes sign is split at its zero, so the areas A+ (D > 0) and A- (D < 0) and
    the lengths of {L_B < L_A} and {L_A < L_B} are exact for any number of crossings.
    Z = min(A+, A-) / (A+ + A-) and Gamma = the smaller of the two lengths; both are 0 under
    dominance in either direction and for coincident curves.
    """
    b_a = np.atleast_2d(np.asarray(b_a, dtype=float))
    b_b = np.atleast_2d(np.asarray(b_b, dtype=float))
    if b_a.shape != b_b.shape:
        raise ValueError(f"b_a and b_b must have the same shape, got {b_a.shape} and {b_b.shape}")
    p = population_weights(p, b_a.shape[1])
    if not (np.all(np.isfinite(b_a)) and np.all(np.isfinite(b_b))):
        raise ValueError("burdens must be finite (no NaN or infinity)")
    if np.any(b_a < 0) or np.any(b_b < 0):
        raise ValueError("burdens must be non-negative")
    if np.any(b_a @ p <= 0) or np.any(b_b @ p <= 0):
        raise ValueError("every burden vector must have a positive total")

    x_a, y_a = lorenz_curve(b_a, p)
    x_b, y_b = lorenz_curve(b_b, p)
    t = np.sort(np.hstack([x_a, x_b[:, 1:-1]]), axis=1)  # 0, every interior breakpoint of both curves, 1
    d = _evaluate(x_a, y_a, t) - _evaluate(x_b, y_b, t)
    d[np.abs(d) < tol] = 0.0

    left, right = t[:, :-1], t[:, 1:]
    d_left, d_right = d[:, :-1], d[:, 1:]
    crosses = d_left * d_right < 0
    zero = np.where(crosses, left + (right - left) * d_left / np.where(crosses, d_left - d_right, 1.0), right)
    # each segment becomes two pieces of constant sign, [left, zero] and [zero, right];
    # the second has length 0 unless the segment crosses
    d_zero = np.where(crosses, 0.0, d_right)
    lengths = np.hstack([zero - left, right - zero])
    areas = np.hstack([(zero - left) * (d_left + d_zero) / 2, (right - zero) * d_right / 2])
    positive, negative = areas > 0, areas < 0

    a_plus = np.where(positive, areas, 0.0).sum(axis=1)
    a_minus = -np.where(negative, areas, 0.0).sum(axis=1)
    length_b_below = np.where(positive, lengths, 0.0).sum(axis=1)  # {L_B < L_A}
    length_a_below = np.where(negative, lengths, 0.0).sum(axis=1)  # {L_A < L_B}

    total = a_plus + a_minus
    zheng = np.divide(np.minimum(a_plus, a_minus), total, out=np.zeros_like(total), where=total > 0)
    ldi = np.minimum(length_a_below, length_b_below)
    return zheng, ldi


# ---------------------------------------------------------------- unit tests
def _run_tests(tol: float = DEFAULT_TOL) -> None:
    """Unit tests of lorenz_measures. Raises AssertionError at the first failure."""
    equal_3, equal_4 = np.full(3, 1 / 3), np.full(4, 1 / 4)
    cases = [  # name, burdens A, burdens B, p, expected Z, expected Gamma (worked out by hand)
        ("dominance, A over B", (1, 1, 1, 1), (4, 1, 3, 2), equal_4, 0.0, 0.0),
        ("dominance, B over A", (1, 6, 6, 7), (5, 5, 5, 5), equal_4, 0.0, 0.0),
        ("dominance, curves touching at a breakpoint", (1, 3, 5), (2, 2, 5), equal_3, 0.0, 0.0),
        ("single crossing", (4, 1, 4), (5, 1.5, 2.5), equal_3, 2 / 7, 4 / 9),
        ("multiple crossings (two)", (12, 2, 14, 12), (13.5, 3, 13.5, 10), equal_4, 7 / 20, 7 / 24),
        ("coincident curves (B = 2 x permuted A)", (1, 2, 3), (6, 2, 4), equal_3, 0.0, 0.0),
    ]
    print("Unit tests of lorenz_measures (each case also with A and B swapped):")
    for name, b_a, b_b, p, z_expected, g_expected in cases:
        for order, (first, second) in (("A, B", (b_a, b_b)), ("B, A", (b_b, b_a))):
            z, g = lorenz_measures([first], [second], p, tol)
            assert abs(z[0] - z_expected) < 1e-12 and abs(g[0] - g_expected) < 1e-12, (
                f"{name} ({order}): got Z={z[0]}, Gamma={g[0]}; expected Z={z_expected}, Gamma={g_expected}")
        print(f"  ok  {name}: Z={z_expected:.4f}, Gamma={g_expected:.4f}")

    # random pairs with unequal weights: swapping, and agreement with a fine-grid computation
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.3, 0.15, 0.45])
    b_a, b_b = rng.random((2, 200, 4))
    z, g = lorenz_measures(b_a, b_b, p, tol)
    z_swap, g_swap = lorenz_measures(b_b, b_a, p, tol)
    assert np.array_equal(z, z_swap) and np.array_equal(g, g_swap), "swapping A and B changed the measures"
    print(f"  ok  swapping A and B on {len(z)} random pairs with unequal weights: identical results")

    grid = np.linspace(0.0, 1.0, 100_001)
    trapezoid = lambda f: ((f[:, 1:] + f[:, :-1]) / 2 * np.diff(grid)).sum(axis=1)
    z_grid, g_grid, n_crossings = (np.empty(len(b_a)) for _ in range(3))
    for start in range(0, len(b_a), 10):  # 10 pairs at a time keeps the memory low
        rows = slice(start, start + 10)
        points = np.broadcast_to(grid, (len(b_a[rows]), grid.size))
        d = _evaluate(*lorenz_curve(b_a[rows], p), points) - _evaluate(*lorenz_curve(b_b[rows], p), points)
        a_plus, a_minus = trapezoid(np.maximum(d, 0)), trapezoid(np.maximum(-d, 0))
        z_grid[rows] = np.minimum(a_plus, a_minus) / (a_plus + a_minus)
        g_grid[rows] = np.minimum((d < 0).mean(axis=1), (d > 0).mean(axis=1))
        signs = np.sign(d[:, 1:-1])
        n_crossings[rows] = (signs[:, 1:] * signs[:, :-1] < 0).sum(axis=1)
    assert np.max(np.abs(z - z_grid)) < 1e-4 and np.max(np.abs(g - g_grid)) < 1e-4, (
        f"fine-grid check failed: max |dZ| = {np.max(np.abs(z - z_grid)):.1e}, max |dGamma| = {np.max(np.abs(g - g_grid)):.1e}")
    print(f"  ok  same {len(z)} pairs against a {grid.size:,}-point grid ({np.sum(n_crossings >= 2)} with 2+ crossings): "
          f"max |dZ| = {np.max(np.abs(z - z_grid)):.1e}, max |dGamma| = {np.max(np.abs(g - g_grid)):.1e}")
    print("All unit tests passed.\n")


# ------------------------------------------------------------ classification
def lorenz_classification(
    path: str | Path,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    test: bool = False,
    verbose: bool = True,
) -> Path:
    """Add the columns 'zheng' and 'LDI' to a pairs file written by sampling.py.

    The file has two rows per pair sharing the same 'id' (the pair number) and burden columns
    b1, ..., bK; the population weights p are read from its metadata. Rows are matched by id, so
    their order in the file does not matter; within a pair, the first row is A. Both rows of a
    pair get the pair's values, and the row order of the file is kept.
    Columns already called 'zheng' or 'LDI' are replaced. The file is rewritten through a temporary
    file, so it is never left half-written.

    With test=True the unit tests run first, and the file is not touched if one fails.
    With verbose=True a progress bar and a run summary are shown.

    Returns the path of the updated file.
    """
    if test:
        _run_tests(tol)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    chunk_size = integer(chunk_size, "chunk_size")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    size_before = path.stat().st_size
    with RunMonitor() as monitor:
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        if b"sampling" not in metadata:
            raise ValueError(f"{path.name} has no 'sampling' metadata: it was not written by sampling.py")
        settings = json.loads(metadata[b"sampling"])
        K, p = settings["K"], np.asarray(settings["p"], dtype=float)
        if "id" not in table.column_names or not pa.types.is_integer(table.schema.field("id").type):
            raise ValueError(f"{path.name} has no integer 'id' column (pair number): it uses an older format, "
                             f"sample it again with sampling.py")
        names = [f"b{k}" for k in range(1, K + 1)]
        missing = [name for name in names if name not in table.column_names]
        if missing:
            raise ValueError(f"{path.name} lacks the burden columns {missing}")
        ids = table["id"].to_numpy()
        order = np.argsort(ids, kind="stable")  # the two rows of each pair side by side, in file order
        sorted_ids = ids[order]
        if len(ids) % 2 or np.any(sorted_ids[0::2] != sorted_ids[1::2]) or np.any(np.diff(sorted_ids[0::2]) == 0):
            raise ValueError(f"{path.name}: every id (pair number) must appear in exactly two rows")
        b = np.column_stack([table[name].to_numpy() for name in names])[order]
        b_a, b_b = b[0::2], b[1::2]
        n = len(b_a)
        monitor.stage("reading parquet")

        if verbose:
            print(f"Lorenz descriptors for {n:,} pairs, K={K}, from {path}")
        zheng, ldi = np.empty(n), np.empty(n)
        with tqdm(total=n, unit="pairs", unit_scale=True, desc="Computing zheng and LDI", disable=not verbose) as bar:
            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                zheng[start:stop], ldi[start:stop] = lorenz_measures(b_a[start:stop], b_b[start:stop], p, tol)
                bar.update(stop - start)
        monitor.stage("computing zheng and LDI")

        replaced = [name for name in MEASURE_COLUMNS if name in table.column_names]
        table = table.drop_columns(replaced)
        # both rows of a pair get the pair's values, in the row order of the file
        zheng_rows, ldi_rows = np.empty(2 * n), np.empty(2 * n)
        zheng_rows[order], ldi_rows[order] = np.repeat(zheng, 2), np.repeat(ldi, 2)
        table = table.append_column("zheng", pa.array(zheng_rows)).append_column("LDI", pa.array(ldi_rows))
        metadata[b"lorenz"] = json.dumps({
            "columns": {"zheng": "min(A+, A-) / (A+ + A-)", "LDI": "min(len{L_A < L_B}, len{L_B < L_A})"},
            "A": "first row of each pair", "B": "second row of each pair", "rows": "same value in both rows of a pair",
            "tol": tol,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        table = table.replace_schema_metadata(metadata)
        temporary = path.with_name(path.name + ".tmp")
        try:
            pq.write_table(table, temporary, row_group_size=2 * chunk_size, use_dictionary=False)
            os.replace(temporary, path)  # fails if another program (e.g. a parquet viewer) keeps the file open
        finally:
            temporary.unlink(missing_ok=True)
        monitor.stage("writing parquet")

        if verbose:
            zheng_zero, ldi_zero = np.count_nonzero(zheng == 0), np.count_nonzero(ldi == 0)
            print_rows([
                ("file", str(path)),
                ("pairs", f"{n:,} ({2 * n:,} rows)"),
                ("columns", "zheng, LDI " + ("replaced" if replaced else "added") + " (same value in both rows of a pair)"),
                ("zheng = 0", f"{zheng_zero:,} pairs ({100 * zheng_zero / n:.2f}%): dominance or coincident curves"),
                ("LDI = 0", f"{ldi_zero:,} pairs ({100 * ldi_zero / n:.2f}%)"),
                ("zheng > 0", f"mean {zheng[zheng > 0].mean():.3f}, max {zheng.max():.3f}" if zheng_zero < n else "no pairs"),
                ("LDI > 0", f"mean {ldi[ldi > 0].mean():.3f}, max {ldi.max():.3f}" if ldi_zero < n else "no pairs"),
                *usage_rows(monitor, n),
                ("file size", f"{fmt_bytes(size_before)} -> {fmt_bytes(path.stat().st_size)}"),
            ])
    return path


# --------------------------------------------------------------------- main
def _run_classification(args: argparse.Namespace) -> Path:
    return lorenz_classification(args.path, args.tol, args.chunk_size, args.test, not args.quiet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lorenz-curve descriptors of the policy pairs.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True, title="what to run")

    classification = subcommands.add_parser(
        "classification", help="add the columns zheng and LDI to a pairs file",
        description="Compute the symmetric Zheng ratio and Lorenz dominance index of every pair and add them "
                    "to the file as the columns 'zheng' and 'LDI'.")
    classification.add_argument("path", type=Path, help="parquet file written by sampling.py")
    classification.add_argument("--tol", type=float, default=DEFAULT_TOL,
                                help="differences between the curves below this are treated as 0")
    classification.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                                help="pairs per computation step and per parquet row group")
    classification.add_argument("--test", action="store_true", help="run the unit tests first")
    classification.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    classification.set_defaults(run=_run_classification)

    args = parser.parse_args()
    path = args.run(args)
    if args.quiet:
        print(f"Added zheng and LDI to {path}")


if __name__ == "__main__":
    main()
