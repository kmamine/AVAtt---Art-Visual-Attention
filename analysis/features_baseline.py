"""Which image features attract fixations? Fixated vs other-paintings baseline (cancels centre bias).

Usage (from the repository root):  python analysis/features_baseline.py   (after composition_detect.py)
Writes figures 28-29, results_features.json and per_image_feature_auc.csv.

For each painting and feature map, AUC = P(feature at an observed fixation > feature at a baseline
fixation), where baseline fixations come from other paintings at the same painting coordinates.
AUC > 0.5: the feature attracts gaze beyond what viewing biases explain. Maps are computed on the
painting crop; local measures use a 1-degree window.
"""
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from analysis import INK, INK2, OUT, SURFACE, plt, save
from paintfix import auc, baseline, bh, ci95, crop_image, load_all, read_map, sample_map

WINDOWS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 15)]
FEATURES = {
    "luminance": "Luminance (L*)",
    "saturation": "Saturation",
    "contrast": "Local RMS contrast",
    "edges": "Edge density",
    "spatial_info": "Local spatial information",
    "colourfulness": "Local colourfulness",
    "salient_object": "Salient-object probability",
    "nearness": "Nearness (inverse depth)",
    "centre_distance": "Distance from painting centre",
}


def box_mean(x, k):
    return cv2.blur(x, (k, k))


def feature_maps(det, name, ppd):
    bgr = crop_image(det, name)
    k = max(3, int(round(ppd)) | 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sob = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    def local_std(x):
        m = box_mean(x, k)
        return np.sqrt(np.maximum(box_mean(x * x, k) - m * m, 0))

    rg, yb = r - g, 0.5 * (r + g) - b
    colourful = np.hypot(local_std(rg), local_std(yb)) + 0.3 * np.hypot(box_mean(np.abs(rg), k), box_mean(np.abs(yb), k))
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    maps = {
        "luminance": L,
        "saturation": hsv[..., 1],
        "contrast": local_std(L) / (box_mean(L, k) + 1.0),
        "edges": cv2.GaussianBlur(sob, (0, 0), ppd / 2),
        "spatial_info": local_std(sob),
        "colourfulness": colourful,
        "centre_distance": np.hypot((xx - w / 2), (yy - h / 2)) / np.hypot(w / 2, h / 2),
    }
    sod = read_map("sod", name)
    if sod is not None:
        maps["salient_object"] = sod.astype(np.float32)
    dep = read_map("depth", name)
    if dep is not None:
        maps["nearness"] = dep.astype(np.float32)
    return maps


def per_painting(job):
    name, det_d, obs, base, ppd = job
    maps = feature_maps({name: det_d}, name, ppd)
    on, bon = obs[obs.on_painting], base[base.on_painting]
    res = {"image": name}
    for feat, m in maps.items():
        res[feat] = auc(sample_map(m, on.u, on.v), sample_map(m, bon.u, bon.v))
        for a, b in WINDOWS:
            ow, bw = on[(on.t_s >= a) & (on.t_s < b)], bon[(bon.t_s >= a) & (bon.t_s < b)]
            res[f"{feat}@{a}-{b}"] = auc(sample_map(m, ow.u, ow.v), sample_map(m, bw.u, bw.v)) if len(ow) > 5 else np.nan
    return res


def main():
    det, fix, ppd = load_all()
    nearness_flip = depth_direction(det)
    names = sorted(det)
    from paintfix import BASE_COLS
    jobs = [(n, det[n], fix.loc[fix.image == n, BASE_COLS], baseline(fix, n, seed=i), ppd) for i, n in enumerate(names)]
    with Pool(min(6, max(1, os.cpu_count() - 2))) as pool:
        rows = pool.map(per_painting, jobs)
    df = pd.DataFrame(rows).set_index("image")
    if nearness_flip:
        df[[c for c in df if c.startswith("nearness")]] = 1 - df[[c for c in df if c.startswith("nearness")]]
    df.round(4).to_csv(os.path.join(OUT, "per_image_feature_auc.csv"))

    feats = [f for f in FEATURES if f in df]
    summ = pd.DataFrame({"mean_auc": df[feats].mean(), "ci": [ci95(df[f]) for f in feats],
                         "p": [stats.wilcoxon(df[f].dropna() - 0.5).pvalue for f in feats],
                         "n": df[feats].notna().sum()})
    summ["q"] = bh(summ.p.to_numpy())
    summ = summ.sort_values("mean_auc")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2a78d6" if v > 0.5 else "#eb6834" for v in summ.mean_auc]
    ax.barh(range(len(summ)), summ.mean_auc - 0.5, left=0.5, xerr=summ.ci, color=colors, height=0.65,
            error_kw={"ecolor": INK2, "elinewidth": 1})
    for i, (f, rrow) in enumerate(summ.iterrows()):
        x = rrow.mean_auc + (rrow.ci if rrow.mean_auc > 0.5 else -rrow.ci)
        ax.text(x + (0.004 if rrow.mean_auc > 0.5 else -0.004), i, f"{rrow.mean_auc:.3f}" + ("*" if rrow.q < 0.05 else ""),
                va="center", ha="left" if rrow.mean_auc > 0.5 else "right", fontsize=9, color=INK2)
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_yticks(range(len(summ)))
    ax.set_yticklabels([FEATURES[f] for f in summ.index])
    lo = min(0.5 - (summ.mean_auc - 0.5).abs().max() - 0.05, 0.4)
    ax.set_xlim(lo, 1 - lo)
    ax.set_xlabel("AUC: fixated vs other-paintings baseline (0.5 = no effect; * q < 0.05)")
    ax.set_title(f"Which image features attract the eye? (n = {len(df)} paintings, mean ± 95% CI)")
    ax.grid(axis="y", visible=False)
    save(fig, "28_feature_auc.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    top = list(summ.index[::-1][:8])
    xs = [np.mean(w) for w in WINDOWS]
    for f, c in zip(top, palette):
        m = [df[f"{f}@{a}-{b}"].mean() for a, b in WINDOWS]
        e = [ci95(df[f"{f}@{a}-{b}"]) for a, b in WINDOWS]
        ax.errorbar(xs, m, yerr=e, color=c, marker="o", ms=4, capsize=0, label=FEATURES[f])
    ax.axhline(0.5, color=INK, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}-{b} s" for a, b in WINDOWS])
    ax.set_ylabel("AUC vs baseline")
    ax.set_title("Feature effects over viewing time")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "29_feature_auc_over_time.png")

    out = {"nearness_inverted_from_depth": bool(nearness_flip),
           "features": json.loads(summ.round(5).to_json(orient="index")),
           "over_time": {f: {f"{a}-{b}s": round(float(df[f"{f}@{a}-{b}"].mean()), 4) for a, b in WINDOWS} for f in feats}}
    with open(os.path.join(OUT, "results_features.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(summ.round(4).to_string())


def depth_direction(det):
    """True if the YOLO26 depth map is 'far = high' (so nearness must be inverted).
    Heuristic: sky (ADE20K) should be far; compare depth inside vs outside sky across paintings."""
    diffs = []
    for name, d in det.items():
        dep, sem = read_map("depth", name), read_map("semantic", name)
        sky_ids = [int(k) for k, v in d.get("semantic_names", {}).items() if v == "sky"]
        if dep is None or sem is None or not sky_ids:
            continue
        sky = np.isin(sem, sky_ids)
        if 0.05 < sky.mean() < 0.9:
            diffs.append(dep[sky].mean() - dep[~sky].mean())
    return bool(np.median(diffs) > 0) if diffs else False


if __name__ == "__main__":
    main()
