"""Helpers shared by the other modules: input checks and the run summary (time and memory).

This module imports nothing from the project, so every other module can import it.
"""

import operator
import sys
import time
import tracemalloc

import numpy as np
import psutil
from numpy.typing import ArrayLike

WEIGHT_SUM_TOL = 1e-9  # population weights must sum to 1 within this tolerance


# --------------------------------------------------------------- input checks
def population_weights(p: ArrayLike | None, K: int) -> np.ndarray:
    """Return the population weights: 1/K for every group when p is None."""
    if p is None:
        return np.full(K, 1.0 / K)
    p = np.asarray(p, dtype=float)
    if p.shape != (K,):
        raise ValueError(f"p must have K={K} entries, got shape {p.shape}")
    if np.any(p <= 0):
        raise ValueError(f"p must be strictly positive; non-positive entries at indices {np.flatnonzero(p <= 0).tolist()}")
    if abs(p.sum() - 1) > WEIGHT_SUM_TOL:
        raise ValueError(f"p must sum to 1 (tolerance {WEIGHT_SUM_TOL}), got {p.sum()}")
    return p


def integer(value, name: str) -> int:
    """value as an int; TypeError for floats, None and other non-integers."""
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None


# ---------------------------------------------------------------- run summary
def fmt_bytes(n_bytes: float | None) -> str:
    if n_bytes is None:
        return "n/a"
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n_bytes >= size:
            return f"{n_bytes / size:.1f} {unit}"
    return f"{n_bytes:.0f} B"


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds < 60 else f"{seconds // 60:.0f} min {seconds % 60:.0f} s"


def peak_process_memory() -> int | None:
    """Peak RAM of this Python process since it started."""
    peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # Windows
    if peak is None:
        try:
            import resource  # Linux / macOS
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            pass
    return peak


class RunMonitor:
    """Wall-clock time per stage, total CPU time and peak memory of one run.

    Use it as a context manager: memory tracing is switched off at the end of the block even if
    the run stops with an error.
    """

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._usage: dict | None = None
        self._owns_tracing = not tracemalloc.is_tracing()
        if self._owns_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        self._start = self._last = time.perf_counter()
        self._cpu_start = time.process_time()

    def __enter__(self) -> "RunMonitor":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def stage(self, name: str) -> None:
        """Record the time since the previous stage ended under name."""
        now = time.perf_counter()
        self.stages[name] = now - self._last
        self._last = now

    def stop(self) -> dict:
        """End the run (only the first call counts) and return its time and memory usage."""
        if self._usage is None:
            peak_traced = tracemalloc.get_traced_memory()[1]
            if self._owns_tracing:
                tracemalloc.stop()
            self._usage = {
                "wall": time.perf_counter() - self._start,
                "cpu": time.process_time() - self._cpu_start,
                "peak_traced": peak_traced,
                "peak_process": peak_process_memory(),
            }
        return self._usage


def usage_rows(monitor: RunMonitor, n_items: int, unit: str = "pairs") -> list[tuple[str, str]]:
    """Stop the monitor and return the time and memory rows of a run summary."""
    usage = monitor.stop()
    return [
        *[(f"time: {name}", fmt_seconds(seconds)) for name, seconds in monitor.stages.items()],
        ("time: total", f"{fmt_seconds(usage['wall'])} wall clock, {fmt_seconds(usage['cpu'])} CPU, "
                        f"{n_items / usage['wall']:,.0f} {unit}/s"),
        ("peak RAM: this run", f"{fmt_bytes(usage['peak_traced'])} (NumPy and Python allocations)"),
        ("peak RAM: process", f"{fmt_bytes(usage['peak_process'])} (whole Python process since it started)"),
    ]


def print_rows(rows: list[tuple[str, str]], title: str = "Run summary") -> None:
    """Print (label, value) rows as an aligned table under title."""
    width = max(len(label) for label, _ in rows)
    print(title)
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
