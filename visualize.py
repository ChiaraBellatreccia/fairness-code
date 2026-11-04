"""Figures.

visualize_lorenz_scatter: the pairs of a pairs file on the plane (zheng, LDI), after lorenz.py has added
those two columns.
visualize_disagreement_scatter: the same plane, one row of three panels per beta, with the pairs colored by
the disagreement between two normalized metrics (GINI-Theil, GINI-bad Atkinson, Theil-bad Atkinson), after
metrics.py has added the normalized metrics; see pairwise_disagreement.
visualize_disagreement_plot: mean disagreement of the bad Atkinson index with GINI and with Theil against
epsilon = -beta.

Terminal: the first argument chooses the figure, followed by its options. The figure is saved in plots/.
    python visualize.py lorenz-scatter results/seed42_K5_n250000_b.parquet
    python visualize.py disagreement-scatter results/seed42_K5_n250000_b.parquet --n-betas 10
    python visualize.py disagreement-plot results/seed42_K5_n250000_b.parquet
    python visualize.py disagreement-scatter --help
Notebook: the figure is shown.
    from visualize import visualize_lorenz_scatter
    visualize_lorenz_scatter("results/seed42_K5_n250000_b.parquet")

Every call ends with a summary of time and memory (verbose=False or --quiet turns it off).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from utils import RunMonitor, fmt_bytes, integer, print_rows, usage_rows

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_POINT_SIZE = 1.0
DEFAULT_ALPHA = 0.2
DPI = 150

DEFAULT_N_BETAS = 10  # rows of the disagreement figure
DEFAULT_TAU = 1e-9  # |difference| of a normalized metric at or below this is a practical tie
AGREEMENT, DISAGREEMENT, TIE = "agreement", "disagreement", "tie"
# D = 0 in grey, D > 0 on a color map
NO_DISAGREEMENT_COLOR, DISAGREEMENT_CMAP = "#c4c4c4", "plasma"
N_COLOR_LEVELS = 64  # disagreement is drawn in this many color bands


def visualize_lorenz_scatter(
    path: str | Path,
    save_to: str | Path | None = None,
    point_size: float = DEFAULT_POINT_SIZE,
    alpha: float = DEFAULT_ALPHA,
    show: bool = True,
    verbose: bool = True,
) -> plt.Figure | None:
    """Scatter plot of the pairs: Zheng ratio (x) against Lorenz dominance index (y).

    path must be a pairs file to which lorenz.py has added the columns 'zheng' and 'LDI'. Both rows
    of a pair (same id) hold the same values, so one point per pair is drawn.
    The figure is saved to save_to when given. With show=True it is shown and None is returned;
    with show=False it is returned (e.g. to change or save it later) without being shown.
    With verbose=True a summary of time and memory is printed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    schema = pq.read_schema(path)
    if "id" not in schema.names or not pa.types.is_integer(schema.field("id").type):
        raise ValueError(f"{path.name} has no integer 'id' column (pair number): it uses an older format, "
                         f"sample it again with sampling.py")
    missing = [name for name in ("zheng", "LDI") if name not in schema.names]
    if missing:
        raise ValueError(f"{path.name} has no column {', '.join(missing)}: "
                         f"run  python lorenz.py classification {path}  first")

    with RunMonitor() as monitor:
        table = pq.read_table(path, columns=["id", "zheng", "LDI"])
        first_row_of_pair = np.unique(table["id"].to_numpy(), return_index=True)[1]
        zheng, ldi = table["zheng"].to_numpy()[first_row_of_pair], table["LDI"].to_numpy()[first_row_of_pair]
        n = len(zheng)
        at_origin = np.count_nonzero((zheng == 0) & (ldi == 0))
        monitor.stage("reading parquet")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(zheng, ldi, s=point_size, alpha=alpha, color="black", linewidths=0, rasterized=True)
        ax.set_xlim(-0.01, 0.51)
        ax.set_ylim(-0.01, 0.51)
        ax.set_aspect("equal")
        ax.set_xlabel("Zheng ratio  Z = min(A+, A-) / (A+ + A-)")
        ax.set_ylabel("Lorenz dominance index  LDI")
        ax.set_title(f"{path.stem}\n{n:,} pairs, {at_origin:,} at (0, 0): dominance or coincident curves",
                     fontsize=10)
        fig.tight_layout()
        monitor.stage("building the figure")

        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, dpi=DPI)
            monitor.stage("drawing and saving the image")

        if verbose:
            print_rows([
                ("data", str(path)),
                ("points", f"{n:,} pairs, {at_origin:,} of them at (0, 0)"),
                ("image", f"{save_to} ({fmt_bytes(save_to.stat().st_size)})" if save_to is not None else "not saved"),
                *usage_rows(monitor, n, unit="points"),
            ])
    if show:
        plt.show()
        return None
    return fig


# ------------------------------------------------------- metric disagreement
def pairwise_disagreement(delta_j: np.ndarray, delta_k: np.ndarray,
                          tau: float = DEFAULT_TAU) -> tuple[np.ndarray, np.ndarray]:
    """Disagreement magnitude and status of two normalized metrics j and k on each policy pair (A, B).

    delta_j, delta_k: differences M_j(A) - M_j(B) and M_k(A) - M_k(B) of two normalized metrics (values
    in [0, 1], higher = more unequal), one entry per pair.

    D_jk = min(|delta_j|, |delta_k|) when the two metrics rank A and B in opposite directions
    (delta_j * delta_k < 0) and both differences exceed tau, and 0 otherwise; D_jk is in [0, 1], and
    larger means a stronger opposite ranking.
    status: 'tie' when |delta_j| <= tau or |delta_k| <= tau (a metric cannot tell A and B apart),
    otherwise 'disagreement' (opposite signs) or 'agreement' (same sign). D_jk = 0 for both agreement
    and ties; the status tells them apart.

    Returns (D_jk, status), two arrays of the same shape as the inputs.
    """
    delta_j, delta_k = np.asarray(delta_j, dtype=float), np.asarray(delta_k, dtype=float)
    if delta_j.shape != delta_k.shape:
        raise ValueError(f"delta_j and delta_k must have the same shape, got {delta_j.shape} and {delta_k.shape}")
    if not (np.all(np.isfinite(delta_j)) and np.all(np.isfinite(delta_k))):
        raise ValueError("the differences must be finite")
    if not tau >= 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    tie = (np.abs(delta_j) <= tau) | (np.abs(delta_k) <= tau)
    opposite = (delta_j * delta_k < 0) & ~tie
    magnitude = np.where(opposite, np.minimum(np.abs(delta_j), np.abs(delta_k)), 0.0)
    status = np.where(tie, TIE, np.where(opposite, DISAGREEMENT, AGREEMENT))
    return magnitude, status


def _in_notebook() -> bool:
    """True inside a Jupyter kernel (notebook, JupyterLab, VS Code notebook)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


def _pair_rows(ids: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of policy A and policy B of every pair, matched by id (A = first row of the pair in the file)."""
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if len(ids) % 2 or np.any(sorted_ids[0::2] != sorted_ids[1::2]) or np.any(np.diff(sorted_ids[0::2]) == 0):
        raise ValueError(f"{name}: every id (pair number) must appear in exactly two rows")
    return order[0::2], order[1::2]


def _beta_grid_of(path: Path, need_lorenz: bool) -> list[float]:
    """Check that path is a pairs file with the normalized metrics (and zheng, LDI when need_lorenz); return its beta grid."""
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    schema = pq.read_schema(path)
    if "id" not in schema.names or not pa.types.is_integer(schema.field("id").type):
        raise ValueError(f"{path.name} has no integer 'id' column (pair number): it uses an older format, "
                         f"sample it again with sampling.py")
    if need_lorenz and ("zheng" not in schema.names or "LDI" not in schema.names):
        raise ValueError(f"{path.name} has no zheng and LDI columns: run  python lorenz.py classification {path}  first")
    metadata = schema.metadata or {}
    if b"metrics" not in metadata or "GINI_norm" not in schema.names or "Theil_norm" not in schema.names:
        raise ValueError(f"{path.name} has no normalized metrics: run  python metrics.py compute {path}  first")
    grid = json.loads(metadata[b"metrics"])["betas"]
    missing = [f"bad_Atk({beta:g})_norm" for beta in grid if f"bad_Atk({beta:g})_norm" not in schema.names]
    if missing:
        raise ValueError(f"{path.name} lacks {missing}: run  python metrics.py compute {path}  again")
    return grid


def visualize_disagreement_scatter(
    path: str | Path,
    n_betas: int = DEFAULT_N_BETAS,
    tau: float = DEFAULT_TAU,
    save: bool | None = None,
    save_to: str | Path | None = None,
    point_size: float = DEFAULT_POINT_SIZE,
    show: bool = True,
    verbose: bool = True,
) -> plt.Figure | None:
    """Pairs on the plane (zheng, LDI), colored by the disagreement between two normalized metrics.

    One row per beta (n_betas values spread evenly over the beta grid stored in the file, both ends
    included) and three panels per row: GINI vs Theil, GINI vs bad Atkinson(beta), Theil vs bad
    Atkinson(beta), all normalized (columns GINI_norm, Theil_norm, bad_Atk(beta)_norm). GINI vs Theil
    does not depend on beta, so its panel is the same in every row. Each point is a pair (A, B):
    D_jk = 0 in grey (including the Lorenz-dominance pairs, all at (0, 0)), D_jk > 0 (see
    pairwise_disagreement; tau is its tolerance) colored by D_jk on one color scale shared by all
    panels, strongest on top. Panel titles give the share of disagreeing pairs among all pairs and
    among the crossing pairs (zheng > 0): pairs whose Lorenz curves do not cross can never disagree.

    Saving: save=None saves only when not run in a notebook, save=True always, save=False never; the
    file is save_to, by default plots/<file name>_disagreement_scatter.png. With show=True the figure is
    shown and None is returned; with show=False it is returned without being shown. With verbose=True
    a progress bar and a summary with the share of disagreeing pairs and the time and memory are shown.
    """
    path = Path(path)
    n_betas = integer(n_betas, "n_betas")
    grid = _beta_grid_of(path, need_lorenz=True)
    if not 1 <= n_betas <= len(grid):
        raise ValueError(f"n_betas must be between 1 and {len(grid)} (the betas stored in the file), got {n_betas}")
    betas = [grid[i] for i in np.round(np.linspace(0, len(grid) - 1, n_betas)).astype(int)]
    atk_columns = [f"bad_Atk({beta:g})_norm" for beta in betas]
    if save is None:
        save = not _in_notebook()
    if save:
        save_to = Path(save_to) if save_to is not None else PLOTS_DIR / f"{path.stem}_disagreement_scatter.png"

    with RunMonitor() as monitor:
        table = pq.read_table(path, columns=["id", "zheng", "LDI", "GINI_norm", "Theil_norm", *atk_columns])
        rows_a, rows_b = _pair_rows(table["id"].to_numpy(), path.name)
        zheng, ldi = table["zheng"].to_numpy()[rows_a], table["LDI"].to_numpy()[rows_a]
        delta = {}
        for name in ["GINI_norm", "Theil_norm", *atk_columns]:
            values = table[name].to_numpy()
            delta[name] = values[rows_a] - values[rows_b]  # M(A) - M(B) for every pair
        n_pairs = len(zheng)
        monitor.stage("reading parquet")

        panels = []  # (row, column, title, magnitude, disagree)
        for row, (beta, atk) in enumerate(zip(betas, atk_columns)):
            for column, (j, k, title) in enumerate([("GINI_norm", "Theil_norm", "GINI vs Theil"),
                                                    ("GINI_norm", atk, f"GINI vs Atk({beta:g})"),
                                                    ("Theil_norm", atk, f"Theil vs Atk({beta:g})")]):
                magnitude, status = pairwise_disagreement(delta[j], delta[k], tau)
                panels.append((row, column, title, magnitude, status == DISAGREEMENT))
        monitor.stage("computing the disagreement")

        vmax = max(panel[3].max() for panel in panels) or 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap(DISAGREEMENT_CMAP, N_COLOR_LEVELS)
        marker = dict(ls="", marker="o", ms=np.sqrt(point_size), mew=0, rasterized=True)  # area point_size, as in scatter
        dominance = zheng == 0  # all at (0, 0); the Lorenz curves do not cross
        crossing = ~dominance
        progress = tqdm(total=len(panels) + bool(save), unit="panel", desc="Drawing panels", disable=not verbose)

        # fixed layout in inches: automatic layouts and shared axes are slow with this many panels
        panel, gap, title_space, left, bottom, top, colorbar_space = 3.0, 0.25, 0.55, 0.65, 0.55, 1.3, 1.2
        width = left + 3 * panel + 2 * gap + colorbar_space
        height = top + n_betas * (panel + title_space) - title_space + bottom
        fig, axes = plt.subplots(n_betas, 3, figsize=(width, height), squeeze=False, gridspec_kw=dict(
            left=left / width, right=(left + 3 * panel + 2 * gap) / width, bottom=bottom / height,
            top=1 - top / height, wspace=gap / panel, hspace=title_space / panel))
        for row, column, title, magnitude, disagree in panels:
            ax = axes[row, column]
            drawn = ~disagree & crossing
            drawn[np.flatnonzero(~disagree & dominance)[:1]] = True  # dominance pairs all sit at (0, 0): one marker
            ax.plot(zheng[drawn], ldi[drawn], color=NO_DISAGREEMENT_COLOR, **marker)
            # disagreement in N_COLOR_LEVELS color bands (much faster than one color per point), strongest on top
            level = np.minimum((magnitude[disagree] / vmax * N_COLOR_LEVELS).astype(int), N_COLOR_LEVELS - 1)
            x, y = zheng[disagree], ldi[disagree]
            for band in np.unique(level):
                ax.plot(x[level == band], y[level == band], color=cmap(band), **marker)
            ax.set_title(f"{title}\ndisagree {100 * disagree.mean():.2f}% of pairs, "
                         f"{100 * disagree[crossing].mean() if crossing.any() else 0:.2f}% of crossing", fontsize=8)
            ax.set_xlim(-0.01, 0.51)
            ax.set_ylim(-0.01, 0.51)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.25, 0.5])
            ax.set_yticks([0, 0.25, 0.5])
            ax.tick_params(labelbottom=row == n_betas - 1, labelleft=column == 0, labelsize=8)
            if row == n_betas - 1:
                ax.set_xlabel("Zheng ratio Z")
            if column == 0:
                ax.set_ylabel("LDI")
            progress.update()
        colorbar_height = min(6.0, height - top - bottom)
        colorbar_axes = fig.add_axes([(left + 3 * panel + 2 * gap + 0.3) / width,
                                      (height - top - colorbar_height) / height, 0.2 / width, colorbar_height / height])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axes,
                     label="disagreement D > 0 (normalized metrics)")
        fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, color=NO_DISAGREEMENT_COLOR,
                                   label="D = 0: no disagreement (Lorenz-dominance pairs are all at (0, 0))")],
                   loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / height), frameon=False)
        fig.suptitle(f"{path.stem}: {n_pairs:,} pairs", fontsize=11, y=1 - 0.12 / height, va="top")
        monitor.stage("building the figure")

        if save:
            progress.set_description("Saving the image")
            save_to.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to, dpi=DPI)
            progress.update()
            monitor.stage("drawing and saving the image")
        progress.close()

        if verbose:
            shares = {}
            for row, column, title, magnitude, disagree in panels:
                shares.setdefault(row, []).append(f"{100 * disagree.mean():.2f}%")
            print_rows([
                ("data", str(path)),
                ("pairs", f"{n_pairs:,} ({np.count_nonzero(dominance):,} with Lorenz dominance)"),
                *[(f"disagree, beta {beta:g}", "GINI-Theil {}, GINI-Atk {}, Theil-Atk {}".format(*shares[row]))
                  for row, beta in enumerate(betas)],
                ("largest D", f"{vmax:.4f} (top of the color scale)"),
                ("image", f"{save_to} ({fmt_bytes(save_to.stat().st_size)})" if save else "not saved"),
                *usage_rows(monitor, n_pairs * len(panels), unit="points"),
            ])
    if show:
        plt.show()
        return None
    return fig


def visualize_disagreement_plot(
    path: str | Path,
    tau: float = DEFAULT_TAU,
    save: bool | None = None,
    save_to: str | Path | None = None,
    show: bool = True,
    verbose: bool = True,
) -> plt.Figure | None:
    """Disagreement of the bad Atkinson index with GINI and with Theil as a function of epsilon = -beta.

    For every beta of the grid stored in the file, D_jk (see pairwise_disagreement; tau is its
    tolerance) is computed for each pair between GINI_norm and bad_Atk(beta)_norm and between Theil_norm
    and bad_Atk(beta)_norm, and its mean over all pairs is plotted against epsilon = -beta: one line per
    metric pair. The mean over all pairs combines how often the metrics disagree and how strongly
    (mean D = share of disagreeing pairs x mean D of the disagreeing pairs). GINI vs Theil does not
    depend on beta and is drawn as a dashed reference line.

    Saving: save=None saves only when not run in a notebook, save=True always, save=False never; the
    file is save_to, by default plots/<file name>_disagreement_plot.png. With show=True the figure is
    shown and None is returned; with show=False it is returned without being shown. With verbose=True
    a progress bar and a summary (mean D and share of disagreeing pairs per beta, time and memory) are
    shown.
    """
    path = Path(path)
    betas = np.array(_beta_grid_of(path, need_lorenz=False))
    atk_columns = [f"bad_Atk({beta:g})_norm" for beta in betas]
    if save is None:
        save = not _in_notebook()
    if save:
        save_to = Path(save_to) if save_to is not None else PLOTS_DIR / f"{path.stem}_disagreement_plot.png"

    with RunMonitor() as monitor:
        table = pq.read_table(path, columns=["id", "GINI_norm", "Theil_norm", *atk_columns])
        rows_a, rows_b = _pair_rows(table["id"].to_numpy(), path.name)
        difference = lambda name: table[name].to_numpy()[rows_a] - table[name].to_numpy()[rows_b]  # M(A) - M(B)
        delta_gini, delta_theil = difference("GINI_norm"), difference("Theil_norm")
        n_pairs = len(rows_a)
        monitor.stage("reading parquet")

        mean_d = {"GINI": np.empty(betas.size), "Theil": np.empty(betas.size)}
        share = {"GINI": np.empty(betas.size), "Theil": np.empty(betas.size)}
        with tqdm(total=betas.size + bool(save), unit="beta", desc="Computing the disagreement",
                  disable=not verbose) as progress:
            for i, column in enumerate(atk_columns):
                delta_atk = difference(column)
                for name, delta in (("GINI", delta_gini), ("Theil", delta_theil)):
                    magnitude, status = pairwise_disagreement(delta, delta_atk, tau)
                    mean_d[name][i], share[name][i] = magnitude.mean(), np.mean(status == DISAGREEMENT)
                progress.update()
            magnitude, status = pairwise_disagreement(delta_gini, delta_theil, tau)
            gini_theil_mean, gini_theil_share = magnitude.mean(), np.mean(status == DISAGREEMENT)
            monitor.stage("computing the disagreement")

            epsilon = -betas
            order = np.argsort(epsilon)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for name, color in (("GINI", "tab:blue"), ("Theil", "tab:orange")):
                ax.plot(epsilon[order], mean_d[name][order], marker="o", ms=4, color=color,
                        label=f"{name} vs bad Atkinson")
            ax.axhline(gini_theil_mean, ls="--", color="grey", label=r"GINI vs Theil (does not depend on $\varepsilon$)")
            ax.set_xlabel(r"inequality aversion  $\varepsilon = -\beta$")
            ax.set_ylabel("mean disagreement D over all pairs")
            ax.set_ylim(bottom=0)
            ax.set_title(f"{path.stem}: {n_pairs:,} pairs", fontsize=10)
            ax.legend(frameon=False)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            monitor.stage("building the figure")

            if save:
                progress.set_description("Saving the image")
                save_to.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_to, dpi=DPI)
                progress.update()
                monitor.stage("saving the image")

        if verbose:
            print_rows([
                ("data", str(path)),
                ("pairs", f"{n_pairs:,}"),
                ("GINI vs Theil", f"mean D {gini_theil_mean:.5f}, disagree {100 * gini_theil_share:.2f}%"),
                *[(f"epsilon {-beta:g}", f"GINI-Atk: mean D {mean_d['GINI'][i]:.5f}, disagree {100 * share['GINI'][i]:.2f}% | "
                                         f"Theil-Atk: mean D {mean_d['Theil'][i]:.5f}, disagree {100 * share['Theil'][i]:.2f}%")
                  for i, beta in sorted(enumerate(betas), key=lambda item: -item[1])],
                ("image", f"{save_to} ({fmt_bytes(save_to.stat().st_size)})" if save else "not saved"),
                *usage_rows(monitor, n_pairs * (2 * betas.size + 1), unit="pair comparisons"),
            ])
    if show:
        plt.show()
        return None
    return fig


# --------------------------------------------------------------------- main
def _run_lorenz_scatter(args: argparse.Namespace) -> None:
    save_to = args.output or PLOTS_DIR / f"{args.path.stem}_lorenz_scatter.png"
    visualize_lorenz_scatter(args.path, save_to, args.point_size, args.alpha, show=args.show, verbose=not args.quiet)
    if args.quiet:
        print(f"Saved {save_to}")


def _run_disagreement_scatter(args: argparse.Namespace) -> None:
    visualize_disagreement_scatter(args.path, args.n_betas, args.tau, save=True, save_to=args.output,
                                   point_size=args.point_size, show=args.show, verbose=not args.quiet)
    if args.quiet:
        print(f"Saved {args.output or PLOTS_DIR / f'{args.path.stem}_disagreement_scatter.png'}")


def _run_disagreement_plot(args: argparse.Namespace) -> None:
    visualize_disagreement_plot(args.path, args.tau, save=True, save_to=args.output, show=args.show,
                                verbose=not args.quiet)
    if args.quiet:
        print(f"Saved {args.output or PLOTS_DIR / f'{args.path.stem}_disagreement_plot.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures of the experiments.")
    subcommands = parser.add_subparsers(dest="subcommand", required=True, title="which figure")

    scatter = subcommands.add_parser(
        "lorenz-scatter", help="pairs on the plane (zheng, LDI)",
        description="Scatter plot of the pairs: Zheng ratio against Lorenz dominance index.")
    scatter.add_argument("path", type=Path, help="pairs file with the columns zheng and LDI (see lorenz.py)")
    scatter.add_argument("--output", type=Path, default=None,
                         help="image file to write (default: plots/<file name>_lorenz_scatter.png)")
    scatter.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    scatter.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="point opacity, 0 to 1")
    scatter.add_argument("--show", action="store_true", help="also open the figure in a window")
    scatter.add_argument("--quiet", action="store_true", help="no run summary")
    scatter.set_defaults(run=_run_lorenz_scatter)

    disagreement = subcommands.add_parser(
        "disagreement-scatter", help="pairs on (zheng, LDI) colored by the disagreement of two normalized metrics",
        description="One row per beta, three panels per row (GINI vs Theil, GINI vs Atk(beta), Theil vs Atk(beta)): "
                    "pairs on the plane (zheng, LDI) colored by the disagreement of the two normalized metrics.")
    disagreement.add_argument("path", type=Path, help="pairs file with zheng, LDI and the normalized metrics "
                                                      "(see lorenz.py and metrics.py)")
    disagreement.add_argument("--n-betas", type=int, default=DEFAULT_N_BETAS,
                              help="rows: beta values spread evenly over the file's beta grid")
    disagreement.add_argument("--tau", type=float, default=DEFAULT_TAU,
                              help="differences of a normalized metric at or below this count as no difference (D = 0)")
    disagreement.add_argument("--output", type=Path, default=None,
                              help="image file to write (default: plots/<file name>_disagreement_scatter.png)")
    disagreement.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    disagreement.add_argument("--show", action="store_true", help="also open the figure in a window")
    disagreement.add_argument("--quiet", action="store_true", help="no run summary")
    disagreement.set_defaults(run=_run_disagreement_scatter)

    plot = subcommands.add_parser(
        "disagreement-plot", help="mean disagreement of bad Atkinson with GINI and Theil against epsilon",
        description="Mean disagreement D over all pairs between bad Atkinson and GINI, and between bad Atkinson "
                    "and Theil, as a function of epsilon = -beta (normalized metrics).")
    plot.add_argument("path", type=Path, help="pairs file with the normalized metrics (see metrics.py)")
    plot.add_argument("--tau", type=float, default=DEFAULT_TAU,
                      help="differences of a normalized metric at or below this count as no difference (D = 0)")
    plot.add_argument("--output", type=Path, default=None,
                      help="image file to write (default: plots/<file name>_disagreement_plot.png)")
    plot.add_argument("--show", action="store_true", help="also open the figure in a window")
    plot.add_argument("--quiet", action="store_true", help="no progress bar and no run summary")
    plot.set_defaults(run=_run_disagreement_plot)

    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")  # save only, no window
    args.run(args)


if __name__ == "__main__":
    main()
