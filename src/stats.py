"""
stats.py — Wilcoxon and Friedman statistical tests.

Matches methodology from the UniBFS paper (Section IV-D):
  - Wilcoxon signed-rank test: pairwise per-dataset comparison.
  - Friedman test: rank all methods across datasets.

Significance threshold: p < 0.05.
"""

import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from typing import Sequence


def wilcoxon_test(
    accs_a: Sequence[float],
    accs_b: Sequence[float],
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test between two sets of run accuracies."""
    stat, p = wilcoxon(accs_a, accs_b, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05}


def friedman_test(*groups: Sequence[float]) -> dict:
    """Friedman test across 3+ groups (methods) over multiple datasets."""
    stat, p = friedmanchisquare(*groups)
    return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05}
