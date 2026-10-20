"""Fixation maps, saliency maps and temporal (accumulated) saliency maps."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .geometry import CANVAS_H, CANVAS_W, PX_PER_DEGREE


def _xy(fixations) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(fixations, list):
        fixations = pd.concat(fixations, ignore_index=True) if fixations else pd.DataFrame(columns=["x", "y"])
    return fixations["x"].to_numpy(int), fixations["y"].to_numpy(int)


def fixation_map(fixations, shape: tuple[int, int] = (CANVAS_H, CANVAS_W)) -> np.ndarray:
    """Binary map with True at every fixation position. ``fixations``: DataFrame or list of DataFrames."""
    x, y = _xy(fixations)
    ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    m = np.zeros(shape, bool)
    m[y[ok], x[ok]] = True
    return m


def saliency_map(fixations, shape: tuple[int, int] = (CANVAS_H, CANVAS_W), sigma: float = PX_PER_DEGREE,
                 normalize: bool = True) -> np.ndarray:
    """Fixation counts convolved with a Gaussian (default sigma = 1 degree of visual angle)."""
    x, y = _xy(fixations)
    ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    m = np.zeros(shape, np.float64)
    np.add.at(m, (y[ok], x[ok]), 1)
    m = gaussian_filter(m, sigma)
    if normalize and m.max() > 0:
        m /= m.max()
    return m.astype(np.float32)


def temporal_saliency_map(fixations, t_ms: float, shape: tuple[int, int] = (CANVAS_H, CANVAS_W),
                          sigma: float = PX_PER_DEGREE, normalize: bool = True) -> np.ndarray:
    """Attention accumulated up to time ``t_ms``: each fixation weighs the time already spent on it,
    min(max(t - onset, 0), duration)."""
    f = pd.concat(fixations, ignore_index=True) if isinstance(fixations, list) else fixations
    w = np.clip(t_ms - f["onset_ms"].to_numpy(float), 0, f["duration_ms"].to_numpy(float))
    x, y = f["x"].to_numpy(int), f["y"].to_numpy(int)
    ok = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0]) & (w > 0)
    m = np.zeros(shape, np.float64)
    np.add.at(m, (y[ok], x[ok]), w[ok])
    m = gaussian_filter(m, sigma)
    if normalize and m.max() > 0:
        m /= m.max()
    return m.astype(np.float32)
