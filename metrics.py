"""Population-weighted inequality metrics for burden distributions.

Conventions
-----------
p : population weights, strictly positive, summing to 1.
b : burdens, non-negative, same length as p.
b_mean = sum_i p_i * b_i  (population-weighted mean burden).

All metrics are 0 for a perfectly equal distribution; higher means more unequal.
All are scale-invariant: multiplying every b_i by the same constant changes nothing.

bad_atkinson is the Atkinson-type index for burdens (bads) of Sheriff and Maguire (2020), with beta < 0.

gini_norm, theil_norm and bad_atkinson_norm divide each index by its maximum for the weights p (whole
burden on the smallest group), so they are in [0, 1].

compute_metrics adds the columns GINI, Theil and bad_Atk(beta) (one per beta of a uniform grid), and their
normalized versions GINI_norm, Theil_norm and bad_Atk(beta)_norm, to a pairs file written by sampling.py,
one value per row (burden vector).
Terminal:
    python metrics.py compute results/seed42_K5_n250000_b.parquet
    python metrics.py compute results/seed42_K5_n250000_b.parquet --beta-range -5 0 --n-beta-samples 20
    python metrics.py compute --help
Notebook:
    from metrics import compute_metrics; compute_metrics("results/seed42_K5_n250000_b.parquet")
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
from scipy.special import xlogy
from tqdm.auto import tqdm

from utils import WEIGHT_SUM_TOL, RunMonitor, fmt_bytes, integer, print_rows, usage_rows

DEFAULT_BETA_RANGE = (-5.0, 0.0)  # uniform beta grid on [-5, 0); 0 is excluded because beta must be < 0
DEFAULT_N_BETA_SAMPLES = 20
DEFAULT_CHUNK_SIZE = 100_000  # rows per computation step and per parquet row group


def _prepare(p: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate inputs and return (p, b, b_mean)."""
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    if p.ndim != 1 or p.shape != b.shape:
        raise ValueError(f"p and b must be 1-D arrays of equal length (got {p.shape} and {b.shape})")
    if p.size == 0:
        raise ValueError("at least one population bin is required")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(b))):
        raise ValueError("p and b must be finite")
    if np.any(p <= 0):
        bad = np.flatnonzero(p <= 0).tolist()
        raise ValueError(f"population weights must be strictly positive (p_i <= 0 at indices {bad})")
    if np.any(b < 0):
        raise ValueError("burdens must be non-negative")
    total = p.sum()
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"population weights must sum to 1 (got {total:.12g})")

    p = p / total  # rescale so the weights sum to 1 (up to float rounding)
    b_mean = float(p @ b)
    if b_mean <= 0:
        raise ValueError("mean burden is zero: the metrics are undefined")
    return p, b, b_mean


def weighted_mean(p: ArrayLike, b: ArrayLike) -> float:
    """b_mean = sum_i p_i b_i (population-weighted mean burden)."""
    return _prepare(p, b)[2]


def gini(p: ArrayLike, b: ArrayLike) -> float:
    """G = sum_i sum_j p_i p_j |b_i - b_j| / (2 b_mean)."""
    p, b, b_mean = _prepare(p, b)
    abs_diff = np.abs(b[:, None] - b[None, :])
    return float(p @ abs_diff @ p / (2 * b_mean))


def theil(p: ArrayLike, b: ArrayLike) -> float:
    """Theil T: sum_i p_i (b_i / b_mean) ln(b_i / b_mean), with 0 * ln 0 = 0."""
    p, b, b_mean = _prepare(p, b)
    r = b / b_mean
    return float(p @ xlogy(r, r))


def atkinson(p: ArrayLike, b: ArrayLike, epsilon: float) -> float:
    """Atkinson index with inequality aversion epsilon >= 0.

    epsilon != 1: A = 1 - [sum_i p_i b_i^(1 - epsilon)]^(1 / (1 - epsilon)) / b_mean
    epsilon == 1: A = 1 - exp(sum_i p_i ln b_i) / b_mean

    With epsilon >= 1, any bin with zero burden gives A = 1.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be a finite number >= 0 (got {epsilon})")
    p, b, b_mean = _prepare(p, b)
    if epsilon == 0:
        return 0.0

    r = b / b_mean  # scale invariance: work with b_i / b_mean
    has_zero = bool(np.any(r == 0))

    if epsilon == 1:
        if has_zero:
            return 1.0
        return float(1 - np.exp(p @ np.log(r)))

    if epsilon > 1 and has_zero:
        return 1.0

    # sum_i p_i r_i^t - 1 computed as sum_i p_i expm1(t ln r_i), using sum(p) == 1.
    # This stays accurate when epsilon is close to 1 (t close to 0).
    t = 1 - epsilon
    with np.errstate(divide="ignore"):  # ln 0 = -inf is fine here: expm1(-inf) = -1
        log_r = np.log(r)
    s = p @ np.expm1(t * log_r)
    return float(1 - np.exp(np.log1p(s) / t))


def atkinson_multi(p: ArrayLike, b: ArrayLike, epsilons: ArrayLike) -> np.ndarray:
    """Atkinson index at several epsilon values at once; same definition and edge cases as atkinson().

    Returns an array with one value per epsilon, in the order given.
    """
    eps = np.atleast_1d(np.asarray(epsilons, dtype=float)).ravel()
    if not np.all(np.isfinite(eps) & (eps >= 0)):
        raise ValueError(f"epsilon values must be finite numbers >= 0 (got {eps.tolist()})")
    p, b, b_mean = _prepare(p, b)
    r = b / b_mean
    has_zero = bool(np.any(r == 0))
    with np.errstate(divide="ignore"):  # ln 0 = -inf is handled below
        log_r = np.log(r)

    out = np.zeros(eps.size)                          # epsilon == 0 -> 0
    at_one = eps == 1
    out[at_one] = 1.0 if has_zero else 1 - np.exp(p @ log_r)
    other = ~(at_one | (eps == 0))
    t = 1 - eps[other]
    # one row per epsilon: sum_i p_i expm1(t ln r_i), accurate when epsilon is close to 1 (see atkinson)
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.expm1(np.outer(t, log_r)) @ p
        values = 1 - np.exp(np.log1p(s) / t)
    if has_zero:
        values[t < 0] = 1.0                           # epsilon > 1 with a zero burden
    out[other] = values
    return out


def _check_betas(beta: float | ArrayLike) -> np.ndarray:
    betas = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    if betas.size == 0 or not np.all(np.isfinite(betas) & (betas < 0)):
        raise ValueError(f"beta must be a finite number < 0 for burdens (got {betas.tolist()})")
    return betas


def _bad_atkinson_rows(p: np.ndarray, r: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """bad_atkinson of every row of r = b / b_mean at every beta, shape (rows, betas). Inputs already checked.

    With burden shares s_i = p_i r_i (summing to 1), l_i = ln r_i, c = max_i l_i and e = -beta > 0,
    sum_i p_i r_i^(1 + e) = exp(e c) [1 + sum_i s_i expm1(e (l_i - c))], so

        ln(1 + I) = [e c + log1p(sum_i s_i expm1(e (l_i - c)))] / (1 + e).

    Every exponent is <= 0, so nothing overflows; expm1 and log1p keep full precision when beta is close
    to 0, where I itself is close to 0.

    Speed: the betas are visited in increasing e, and expm1(e x) is updated from the previous value with
    expm1(a + b) = expm1(a) (1 + expm1(b)) + expm1(b): on a uniform grid, expm1 of the step is computed
    once and every further beta costs one multiplication and one addition per entry. All terms are <= 0,
    so the update has no cancellation.
    """
    with np.errstate(divide="ignore"):  # ln 0 = -inf: a zero burden has share 0 and contributes nothing
        log_r = np.log(r)
    shares = p * r
    c = log_r.max(axis=1)  # finite: the mean burden is positive
    shifted = log_r - c[:, None]  # <= 0
    log1p_index = np.empty((r.shape[0], betas.size))
    terms = np.empty_like(shifted)  # expm1(e * shifted) for the current e
    step_terms = np.empty_like(shifted)  # expm1(step * shifted)
    growth = np.empty_like(shifted)  # 1 + step_terms
    current_e, current_step = None, None
    for j in np.argsort(-betas):  # increasing e
        e = -betas[j]
        if current_e is None:
            np.multiply(shifted, e, out=terms)
            np.expm1(terms, out=terms)
        elif e > current_e:
            step = e - current_e
            if current_step is None or not np.isclose(step, current_step, rtol=1e-12, atol=0):
                np.multiply(shifted, step, out=step_terms)
                np.expm1(step_terms, out=step_terms)
                np.add(step_terms, 1, out=growth)
                current_step = step
            terms *= growth
            terms += step_terms
        current_e = e
        log1p_index[:, j] = (e * c + np.log1p(np.einsum("ij,ij->i", terms, shares))) / (1 + e)
    return np.expm1(log1p_index)


def bad_atkinson(p: ArrayLike, b: ArrayLike, beta: float) -> float:
    """Atkinson-type relative inequality index for burdens (bads), with inequality aversion beta < 0.

    I = [sum_i p_i (b_i / b_mean)^(1 - beta)]^(1 / (1 - beta)) - 1 = M_(1 - beta)(b) / b_mean - 1,

    where M_q is the population-weighted power mean of order q. Sheriff and Maguire define it for
    undesirable outcomes with beta < 0, so the order 1 - beta is above 1 and the most burdened groups
    weigh more: the more negative beta, the stronger the aversion to inequality. As beta -> 0 the index
    goes to 0 and I / (-beta) -> Theil; as beta -> -infinity, I -> max_i b_i / b_mean - 1.
    This is not the income convention of atkinson() (epsilon >= 0, order 1 - epsilon below 1).

    Zero burdens are allowed: they contribute 0 to the sum.

    Reference: Sheriff, G. and Maguire, K. (2020). Health risk, inequality indexes, and environmental
    justice. Risk Analysis. doi:10.1111/risa.13562
    """
    betas = _check_betas(beta)
    if betas.size != 1:
        raise ValueError("beta must be a single number")
    p, b, b_mean = _prepare(p, b)
    return float(_bad_atkinson_rows(p, (b / b_mean)[None, :], betas)[0, 0])


# ------------------------------------------------------ normalized versions
# Each index is divided by its largest possible value for the weights p, reached when the whole burden
# falls on the smallest group; the normalized index is in [0, 1] and equals 1 there. With K and p fixed
# the divisor is a positive constant (one per beta for bad_atkinson), so every ranking and verdict is the
# same as with the raw index.

def _two_groups(p: np.ndarray) -> np.ndarray:
    if p.size < 2:
        raise ValueError("normalized indices need at least two groups (with one group every index is 0)")
    return p


def _gini_max(p: np.ndarray) -> float:
    """Largest gini() for weights p: 1 - p_min."""
    return 1.0 - p.min()


def _theil_max(p: np.ndarray) -> float:
    """Largest theil() for weights p: ln(1 / p_min)."""
    return -np.log(p.min())


def _bad_atkinson_max(p: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Largest bad_atkinson() for weights p at each beta: p_min^(beta / (1 - beta)) - 1."""
    return np.expm1(betas / (1 - betas) * np.log(p.min()))  # expm1: accurate when beta is close to 0


def gini_norm(p: ArrayLike, b: ArrayLike) -> float:
    """Gini divided by its maximum 1 - p_min (all burden on the smallest group); in [0, 1]."""
    value = gini(p, b)
    return value / _gini_max(_two_groups(_prepare(p, b)[0]))


def theil_norm(p: ArrayLike, b: ArrayLike) -> float:
    """Theil divided by its maximum ln(1 / p_min) (all burden on the smallest group); in [0, 1].

    Theil is the Kullback-Leibler divergence of the burden shares p_i b_i / sum_j p_j b_j from the
    population shares p_i, which is largest when the whole burden is on the smallest group.
    """
    value = theil(p, b)
    return value / _theil_max(_two_groups(_prepare(p, b)[0]))


def bad_atkinson_norm(p: ArrayLike, b: ArrayLike, beta: float) -> float:
    """bad_atkinson divided by its maximum p_min^(beta / (1 - beta)) - 1 at the same beta; in [0, 1].

    The maximum is reached when the whole burden is on the smallest group. As beta -> 0 the normalized
    index tends to theil_norm, so Theil and the bad Atkinson indices stay on one scale.
    """
    value = bad_atkinson(p, b, beta)
    return value / float(_bad_atkinson_max(_two_groups(_prepare(p, b)[0]), np.array([beta]))[0])


def inequality_metrics(p: ArrayLike, b: ArrayLike, epsilon: float, beta: float | None = None) -> dict[str, float]:
    """All three metrics at once, plus bad_atkinson when beta is given."""
    metrics = {
        "gini": gini(p, b),
        "theil": theil(p, b),
        "atkinson": atkinson(p, b, epsilon),
    }
    if beta is not None:
        metrics["bad_atkinson"] = bad_atkinson(p, b, beta)
    return metrics


# ------------------------------------------------- all rows of a pairs file
def _prepare_rows(p: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate weights p (K,) and burden rows b (n, K) as in _prepare; return (p, b, b_mean per row)."""
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    if p.ndim != 1 or b.ndim != 2 or b.shape[1] != p.size:
        raise ValueError(f"p must be 1-D and b 2-D with one column per weight (got {p.shape} and {b.shape})")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(b))):
        raise ValueError("p and b must be finite")
    if np.any(p <= 0):
        raise ValueError(f"population weights must be strictly positive (p_i <= 0 at indices {np.flatnonzero(p <= 0).tolist()})")
    if abs(p.sum() - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"population weights must sum to 1 (got {p.sum():.12g})")
    if np.any(b < 0):
        raise ValueError("burdens must be non-negative")
    p = p / p.sum()
    b_mean = b @ p
    if np.any(b_mean <= 0):
        raise ValueError(f"{np.count_nonzero(b_mean <= 0)} rows have zero mean burden: the metrics are undefined")
    return p, b, b_mean


def _gini_rows(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """gini() of every row of r = b / b_mean, via the sorted form sum_k p_k r_k (2 F_k - p_k - 1), F = cumulative p."""
    order = np.argsort(r, axis=1)
    r_sorted = np.take_along_axis(r, order, axis=1)
    p_sorted = p[order]
    return (p_sorted * r_sorted * (2 * np.cumsum(p_sorted, axis=1) - p_sorted - 1)).sum(axis=1)


def beta_grid(beta_range: tuple[float, float] = DEFAULT_BETA_RANGE,
              n_beta_samples: int = DEFAULT_N_BETA_SAMPLES) -> np.ndarray:
    """Uniform grid of n_beta_samples values on [lo, hi] (on [lo, 0) when hi = 0, as beta must be < 0)."""
    lo, hi = beta_range
    if not (np.isfinite(lo) and lo < hi <= 0):
        raise ValueError(f"beta_range must satisfy lo < hi <= 0, got {beta_range}")
    if n_beta_samples < 1:
        raise ValueError(f"n_beta_samples must be positive, got {n_beta_samples}")
    return np.linspace(lo, hi, n_beta_samples, endpoint=hi < 0)


def compute_metrics(
    path: str | Path,
    beta_range: tuple[float, float] = DEFAULT_BETA_RANGE,
    n_beta_samples: int = DEFAULT_N_BETA_SAMPLES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> Path:
    """Add GINI, Theil and bad_Atk(beta) (one per beta), raw and normalized, to a pairs file written by sampling.py.

    Every row (one burden vector b1, ..., bK) gets its own values; the population weights p are read
    from the file metadata. The betas form a uniform grid on beta_range (see beta_grid). The normalized
    values (see gini_norm, theil_norm, bad_atkinson_norm) go into columns with the same names plus
    '_norm': GINI_norm, Theil_norm, bad_Atk(beta)_norm. Columns from an earlier run (GINI, Theil, their
    _norm versions and all bad_Atk(...) columns) are replaced. The file is rewritten through a temporary
    file, so it is never left half-written. With verbose=True a progress bar and a run summary are shown.

    Returns the path of the updated file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    n_beta_samples, chunk_size = integer(n_beta_samples, "n_beta_samples"), integer(chunk_size, "chunk_size")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    betas = beta_grid(beta_range, n_beta_samples)
    atk_names = [f"bad_Atk({beta:g})" for beta in betas]
    if len(set(atk_names)) < len(atk_names):
        raise ValueError("two beta values get the same column name: use fewer beta samples")

    size_before = path.stat().st_size
    with RunMonitor() as monitor:
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        if b"sampling" not in metadata:
            raise ValueError(f"{path.name} has no 'sampling' metadata: it was not written by sampling.py")
        settings = json.loads(metadata[b"sampling"])
        K = settings["K"]
        names = [f"b{k}" for k in range(1, K + 1)]
        missing = [name for name in names if name not in table.column_names]
        if missing:
            raise ValueError(f"{path.name} lacks the burden columns {missing}")
        p, b, b_mean = _prepare_rows(settings["p"], np.column_stack([table[name].to_numpy() for name in names]))
        n_rows = len(b)
        monitor.stage("reading parquet")

        if verbose:
            print(f"Metrics for {n_rows:,} rows, K={K}: GINI, Theil and bad_Atk at {betas.size} beta values "
                  f"from {betas[0]:g} to {betas[-1]:g}")
        gini_values, theil_values = np.empty(n_rows), np.empty(n_rows)
        atk_values = np.empty((n_rows, betas.size))
        with tqdm(total=n_rows, unit="rows", unit_scale=True, desc="Computing metrics", disable=not verbose) as bar:
            for start in range(0, n_rows, chunk_size):
                rows = slice(start, start + chunk_size)
                r = b[rows] / b_mean[rows, None]  # scale invariance: every metric is a function of b / b_mean
                gini_values[rows] = _gini_rows(p, r)
                theil_values[rows] = xlogy(r, r) @ p
                atk_values[rows] = _bad_atkinson_rows(p, r, betas)
                bar.update(len(r))
        # normalized: each index divided by its maximum for these weights (a constant, one per beta)
        maxima = {"GINI": _gini_max(_two_groups(p)), "Theil": _theil_max(p), "bad_Atk": _bad_atkinson_max(p, betas)}
        gini_norm_values, theil_norm_values = gini_values / maxima["GINI"], theil_values / maxima["Theil"]
        atk_norm_values = atk_values / maxima["bad_Atk"]
        monitor.stage("computing the metrics")

        replaced = [name for name in table.column_names
                    if name in ("GINI", "Theil", "GINI_norm", "Theil_norm") or name.startswith("bad_Atk(")]
        table = table.drop_columns(replaced)
        columns = [("GINI", gini_values), ("Theil", theil_values)]
        columns += [(name, atk_values[:, j]) for j, name in enumerate(atk_names)]
        columns += [("GINI_norm", gini_norm_values), ("Theil_norm", theil_norm_values)]
        columns += [(f"{name}_norm", atk_norm_values[:, j]) for j, name in enumerate(atk_names)]
        for name, values in columns:
            table = table.append_column(name, pa.array(values))
        metadata[b"metrics"] = json.dumps({
            "GINI": "sum_i sum_j p_i p_j |b_i - b_j| / (2 b_mean)",
            "Theil": "sum_i p_i (b_i / b_mean) ln(b_i / b_mean)",
            "bad_Atk(beta)": "[sum_i p_i (b_i / b_mean)^(1 - beta)]^(1 / (1 - beta)) - 1, beta < 0 "
                             "(Sheriff and Maguire 2020, doi:10.1111/risa.13562)",
            "_norm": "index divided by its maximum for these weights (whole burden on the smallest group): "
                     "GINI / (1 - p_min), Theil / ln(1 / p_min), bad_Atk(beta) / (p_min^(beta / (1 - beta)) - 1)",
            "maxima": {"GINI": maxima["GINI"], "Theil": maxima["Theil"], "bad_Atk": maxima["bad_Atk"].tolist()},
            "beta_range": list(beta_range), "n_beta_samples": n_beta_samples, "betas": betas.tolist(),
            "rows": "one value per row (burden vector)",
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
                ("rows", f"{n_rows:,} burden vectors"),
                ("columns", f"GINI, Theil, {atk_names[0]} ... {atk_names[-1]} ({betas.size} betas), "
                            f"each also as _norm, " + ("replaced" if replaced else "added")),
                ("GINI", f"mean {gini_values.mean():.4f}, max {gini_values.max():.4f}"),
                ("Theil", f"mean {theil_values.mean():.4f}, max {theil_values.max():.4f}"),
                ("GINI_norm", f"mean {gini_norm_values.mean():.4f}, max {gini_norm_values.max():.4f} "
                              f"(GINI / {maxima['GINI']:.4g})"),
                ("Theil_norm", f"mean {theil_norm_values.mean():.4f}, max {theil_norm_values.max():.4f} "
                               f"(Theil / {maxima['Theil']:.4g})"),
                *usage_rows(monitor, n_rows, unit="rows"),
                ("file size", f"{fmt_bytes(size_before)} -> {fmt_bytes(path.stat().st_size)}"),
            ])
    return path


# --------------------------------------------------------------------- main
def _run_compute(args: argparse.Namespace) -> Path:
    return compute_metrics(args.path, tuple(args.beta_range), args.n_beta_samples, args.chunk_size, not args.quiet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inequality metrics of the burden vectors in a pairs file.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True, title="what to run")

    compute = subcommands.add_parser(
        "compute", help="add the columns GINI, Theil and bad_Atk(beta), raw and _norm, to a pairs file",
        description="Compute GINI, Theil and bad_Atk(beta) on a uniform beta grid for every row of a pairs file, "
                    "raw and normalized (columns with '_norm' added to the name).")
    compute.add_argument("path", type=Path, help="parquet file written by sampling.py")
    compute.add_argument("--beta-range", type=float, nargs=2, default=DEFAULT_BETA_RANGE, metavar=("LO", "HI"),
                         help="beta grid on [LO, HI]; HI = 0 is excluded, as beta must be < 0 (default -5 0)")
    compute.add_argument("--n-beta-samples", type=int, default=DEFAULT_N_BETA_SAMPLES, help="number of beta values")
    compute.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                         help="rows per computation step and per parquet row group")
    compute.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    compute.set_defaults(run=_run_compute)

    args = parser.parse_args()
    path = args.run(args)
    if args.quiet:
        print(f"Added GINI, Theil and bad_Atk columns, raw and _norm, to {path}")


if __name__ == "__main__":
    main()