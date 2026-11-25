"""Statistical reporting over random seeds (Section 5.2).

Paired t-test between TFRN and each baseline on average accuracy across the
same seeds; Wilcoxon signed-rank as a non-parametric check.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def mean_std(values: Sequence[float]):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def paired_tests(tfrn_avgs: Sequence[float], baseline_avgs: Sequence[float]):
    """Return dict with mean gain (percentage points), t-test p, Wilcoxon p."""
    a, b = np.asarray(tfrn_avgs, float), np.asarray(baseline_avgs, float)
    if len(a) != len(b):
        raise ValueError("seed counts differ")
    diff = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
    except ValueError:            # all differences identical / zero
        w_p = float("nan")
    return dict(delta_pp=float(diff.mean()) * 100.0, ttest_p=t_p, wilcoxon_p=w_p)
