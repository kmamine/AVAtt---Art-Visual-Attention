"""Shared helpers: fixations in painting coordinates and the other-paintings (shuffled) baseline.

Painting coordinates: u = (x - x0) / w, v = (y - y0) / h over the painting crop (grey frame excluded).
Baseline for painting A: fixations made on all OTHER paintings, mapped into A by their (u, v).
This keeps viewing biases (centre bias, reading direction, temporal order) but breaks the link to A's
content, so "observed vs baseline" isolates the effect of the painting itself.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from analysis import DATA, OUT, PX_PER_MM, load

COMP = os.path.join(OUT, "composition")
EPS = 1e-9


def load_all():
    """Fixations in painting coordinates. Uses composition detections when present (crop sizes, faces, ...),
    otherwise only the painting boxes from metadata.csv."""
    path = os.path.join(COMP, "detections.json")
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
    if os.path.exists(path):
        with open(path) as f:
            det = json.load(f)
    else:
        det = {}
    for i, r in meta.iterrows():
        box = [int(r.painting_x0), int(r.painting_y0), int(r.painting_x1), int(r.painting_y1)]
        det.setdefault(i, {}).update({"crop_box": box, "crop_size": [box[2] - box[0], box[3] - box[1]],
                                       "aspect_ratio": round((box[2] - box[0]) / (box[3] - box[1]), 4)})
    fix, _, sp = load()
    ppd = 2 * fix.distance_mm.mean() * np.tan(np.radians(0.5)) * PX_PER_MM
    box = pd.DataFrame({n: d["crop_box"] for n, d in det.items()}, index=["x0", "y0", "x1", "y1"]).T
    fix = fix.join(box, on="image")
    fix["u"] = (fix.x - fix.x0) / (fix.x1 - fix.x0)
    fix["v"] = (fix.y - fix.y0) / (fix.y1 - fix.y0)
    fix["on_painting"] = fix.u.between(0, 1, inclusive="left") & fix.v.between(0, 1, inclusive="left")
    fix["t_s"] = fix.onset_ms / 1000
    fix = fix.sort_values(["image", "observer", "onset_ms"]).reset_index(drop=True)
    fix["fix_idx"] = fix.groupby(["image", "observer"]).cumcount()
    return det, fix, ppd


BASE_COLS = ["image", "observer", "u", "v", "t_s", "on_painting", "duration_ms"]


def baseline(fix, image, n=20000, seed=0):
    """Whole scanpaths from other paintings (about n fixations), mapped into this painting by (u, v).

    Whole scanpaths are kept so saccades and first-fixation times stay valid; use .on_painting to keep
    fixations that fell on their own painting."""
    other = fix[fix.image != image]
    keys = other[["image", "observer"]].drop_duplicates().sample(frac=1, random_state=seed)
    sizes = other.groupby(["image", "observer"]).size().reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    keep = keys.iloc[: int(np.searchsorted(np.cumsum(sizes), n)) + 1]
    out = other.merge(keep, on=["image", "observer"])
    return out[BASE_COLS]


def crop_image(det, name):
    x0, y0, x1, y1 = det[name]["crop_box"]
    return cv2.imread(os.path.join(DATA, "images", name + ".jpg"))[y0:y1, x0:x1]


def read_map(folder, name):
    p = os.path.join(COMP, folder, name + ".png")
    return cv2.imread(p, 0) if os.path.exists(p) else None


def sample_map(m, u, v):
    """Nearest-pixel lookup of a crop-sized map at painting coordinates."""
    h, w = m.shape[:2]
    xi = np.clip((np.asarray(u) * w).astype(int), 0, w - 1)
    yi = np.clip((np.asarray(v) * h).astype(int), 0, h - 1)
    return m[yi, xi]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty_like(q)
    tmp[order] = np.minimum(q, 1)
    out[ok] = tmp
    return out


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
