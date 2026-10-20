"""Saliency metrics (Bylinskii et al., 2019) and inter-observer congruency.

All functions take a predicted saliency map and either a binary fixation map (location-based metrics) or a
ground-truth saliency map (distribution-based metrics) of the same shape.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _norm01(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.float64)
    return (s - s.min()) / (s.max() - s.min() + EPS)


def auc_judd(sal: np.ndarray, fixmap: np.ndarray, jitter: bool = True) -> float:
    """AUC-Judd: thresholds at the saliency of each fixated pixel."""
    s = _norm01(sal).ravel()
    if jitter:
        s = s + np.random.default_rng(0).random(s.size) * 1e-7     # break ties, as in the MIT benchmark code
    f = s[np.flatnonzero(fixmap.ravel())]
    if f.size == 0:
        return float("nan")
    thresholds = np.sort(f)[::-1]
    above = s.size - np.searchsorted(np.sort(s), thresholds, side="left")
    k = np.arange(1, f.size + 1)
    tp = np.r_[0, k / f.size, 1]
    fp = np.r_[0, (above - k) / (s.size - f.size), 1]
    return float(_trapz(tp, fp))


def auc_shuffled(sal: np.ndarray, fixmap: np.ndarray, other_fixmap: np.ndarray) -> float:
    """Shuffled AUC: fixated pixels vs fixations from other images (cancels centre bias)."""
    s = sal.ravel().astype(np.float64)
    pos = s[np.flatnonzero(fixmap.ravel())]
    neg = s[np.flatnonzero(other_fixmap.ravel())]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def nss(sal: np.ndarray, fixmap: np.ndarray) -> float:
    """Normalized scanpath saliency: mean z-scored saliency at fixated pixels."""
    s = sal.astype(np.float64)
    z = (s - s.mean()) / (s.std() + EPS)
    return float(z[fixmap.astype(bool)].mean())


def cc(sal: np.ndarray, gt: np.ndarray) -> float:
    """Pearson correlation between two saliency maps."""
    a, b = sal.ravel().astype(np.float64), gt.ravel().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def sim(sal: np.ndarray, gt: np.ndarray) -> float:
    """Histogram intersection of the two maps normalised to sum 1."""
    a, b = sal.astype(np.float64), gt.astype(np.float64)
    return float(np.minimum(a / (a.sum() + EPS), b / (b.sum() + EPS)).sum())


def kld(sal: np.ndarray, gt: np.ndarray) -> float:
    """KL divergence of the ground truth from the prediction (lower is better)."""
    a, b = sal.astype(np.float64), gt.astype(np.float64)
    a, b = a / (a.sum() + EPS), b / (b.sum() + EPS)
    return float((b * np.log(EPS + b / (a + EPS))).sum())


def all_metrics(sal: np.ndarray, fixmap: np.ndarray, gt: np.ndarray, other_fixmap: np.ndarray | None = None) -> dict:
    """AUC-Judd, sAUC (if ``other_fixmap`` is given), NSS, CC, SIM and KL in one call."""
    out = {"AUC-Judd": auc_judd(sal, fixmap), "NSS": nss(sal, fixmap), "CC": cc(sal, gt), "SIM": sim(sal, gt),
           "KL": kld(sal, gt)}
    if other_fixmap is not None:
        out["sAUC"] = auc_shuffled(sal, fixmap, other_fixmap)
    return out
