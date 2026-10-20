"""Painting-level factors: which properties of a painting shape how it is viewed?

Usage (from the repository root):  python analysis/painting_factors.py
  (after analysis.py, benchmark.py, composition_detect.py, composition_fit.py)
Writes figures 31-32, results_painting_factors.json and per_image_factors.csv.

Outcomes per painting: heatmap entropy, fixation dispersion, human agreement (split-half CC),
inter-observer congruency, mean fixation duration, mean saccade amplitude.
Predictors: colourfulness, spatial information, aspect ratio, on-screen size, faces, focal subjects,
salient-object area, symmetry, horizon, line orientations, composition fits, collection.
Spearman correlations with Benjamini-Hochberg correction; outcomes by CACNet composition class.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from analysis import INK, INK2, OUT, SERIES, SURFACE, plt, save
from paintfix import COMP, bh, load_all

OUTCOMES = {"entropy": "Heatmap entropy (bits)", "dispersion_deg": "Fixation dispersion (deg)",
            "human_CC": "Human agreement (split-half CC)", "IOC": "Inter-observer congruency",
            "fix_dur_ms": "Mean fixation duration (ms)", "sacc_amp_deg": "Mean saccade amplitude (deg)"}
PREDICTORS = {"colorfulness": "Colourfulness", "SI": "Spatial information", "aspect_ratio": "Aspect ratio (w/h)",
              "painting_area_pct": "On-screen size (% of image)", "n_faces": "Number of faces",
              "n_subjects": "Number of focal subjects", "salient_area": "Salient-object area",
              "mirror_symmetry": "Mirror symmetry", "has_horizon": "Horizon present",
              "lines_horizontal": "Horizontal-line share", "lines_vertical": "Vertical-line share",
              "lines_diagonal": "Diagonal-line share", "fit_thirds": "Rule-of-thirds fit score",
              "fit_centre": "Centre fit score", "fit_diagonal": "Diagonal fit score",
              "fit_spiral": "Golden-spiral fit score (stretched)"}


def entropy_map(xs, ys, w, h, sigma):
    m = np.zeros((h, w), np.float32)
    np.add.at(m, (np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)), 1)
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    p = m / m.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    det, fix, ppd = load_all()
    with open(os.path.join(COMP, "fits.json")) as f:
        fits = json.load(f)
    pi = pd.read_csv(os.path.join(OUT, "per_image.csv"), index_col=0)
    ha = pd.read_csv(os.path.join(OUT, "per_image_human_agreement.csv"), index_col=0)
    rows = []
    for name, g in fix.groupby("image"):
        d = det[name]
        w, h = d["crop_size"]
        on = g[g.on_painting]
        xs, ys = (on.u * w).astype(int).to_numpy(), (on.v * h).astype(int).to_numpy()
        sc = 0.25   # entropy at quarter resolution, same sigma in degrees
        ent = entropy_map((xs * sc).astype(int), (ys * sc).astype(int), max(int(w * sc), 1), max(int(h * sc), 1), ppd * sc)
        disp = float(np.hypot(on.u * w - (on.u * w).mean(), on.v * h - (on.v * h).mean()).mean() / ppd)
        dx = g.groupby("observer").x.diff()
        dy = g.groupby("observer").y.diff()
        amp = np.degrees(np.arctan(np.hypot(dx, dy) / 3.6161 / g.distance_mm.fillna(g.distance_mm.mean())))
        orient = d["line_orientation_share"]
        fit = {fam: fits[name][fam].get("score", np.nan) for fam in fits[name]}
        top_cac = max(d["composition_probs"]["cacnet"], key=d["composition_probs"]["cacnet"].get)
        top_clip = max(d["composition_probs"]["clip"], key=d["composition_probs"]["clip"].get)
        rows.append({"image": name, "collection": pi.loc[name, "collection"],
                     "entropy": ent, "dispersion_deg": disp, "human_CC": ha.loc[name, "CC"] if name in ha.index else np.nan,
                     "IOC": pi.loc[name, "IOC"], "fix_dur_ms": g.duration_ms.mean(), "sacc_amp_deg": float(np.nanmean(amp)),
                     "colorfulness": pi.loc[name, "colorfulness"], "SI": pi.loc[name, "SI"],
                     "aspect_ratio": d["aspect_ratio"], "painting_area_pct": pi.loc[name, "painting_area_pct"],
                     "n_faces": len(d["faces"]), "n_subjects": len(d["subjects"]), "salient_area": d["salient_area_frac"],
                     "mirror_symmetry": d["mirror_symmetry"], "has_horizon": float(d["horizon_v"] is not None),
                     "lines_horizontal": orient["horizontal"], "lines_vertical": orient["vertical"],
                     "lines_diagonal": orient["diagonal"], "fit_thirds": fit.get("thirds"), "fit_centre": fit.get("centre"),
                     "fit_diagonal": fit.get("diagonal"), "fit_spiral": fit.get("spiral_stretched"),
                     "cacnet_class": top_cac, "clip_class": top_clip})
    df = pd.DataFrame(rows).set_index("image")
    df.round(5).to_csv(os.path.join(OUT, "per_image_factors.csv"))

    # --- figure 31: Spearman correlation matrix, BH-corrected
    rho = pd.DataFrame(index=list(PREDICTORS), columns=list(OUTCOMES), dtype=float)
    pval = rho.copy()
    for p in PREDICTORS:
        for o in OUTCOMES:
            sub = df[[p, o]].dropna()
            r, pv = stats.spearmanr(sub[p], sub[o]) if sub[p].nunique() > 1 else (np.nan, np.nan)
            rho.loc[p, o], pval.loc[p, o] = r, pv
    q = pd.DataFrame(bh(pval.to_numpy().ravel()).reshape(pval.shape), index=pval.index, columns=pval.columns)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4b393", "#f0efec", "#86b6ef", "#2a78d6"])
    im = ax.imshow(rho.to_numpy(float), cmap=div, vmin=-0.6, vmax=0.6, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.iloc[i, j]
            if np.isnan(v):
                continue
            star = "**" if q.iloc[i, j] < 0.01 else "*" if q.iloc[i, j] < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels(list(OUTCOMES.values()), rotation=30, ha="right")
    ax.set_yticks(range(len(PREDICTORS)))
    ax.set_yticklabels(list(PREDICTORS.values()))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Spearman ρ", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"Painting properties vs viewing behaviour (n = {len(df)}; * q < 0.05, ** q < 0.01, BH)")
    fig.tight_layout()
    save(fig, "31_painting_factors.png")

    # --- figure 32: outcomes by composition class (CACNet)
    classes = df.cacnet_class.value_counts()
    classes = classes[classes >= 5].index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    kw = {}
    for ax, o in zip(axes, ["entropy", "human_CC", "dispersion_deg"]):
        groups = [df.loc[df.cacnet_class == c, o].dropna() for c in classes]
        bp = ax.boxplot(groups, labels=[f"{c}\n(n={len(g)})" for c, g in zip(classes, groups)], showfliers=False,
                        patch_artist=True, medianprops={"color": INK}, boxprops={"facecolor": "#86b6ef", "edgecolor": SURFACE})
        H, p = stats.kruskal(*groups) if len(groups) > 1 else (np.nan, np.nan)
        kw[o] = {"H": float(H), "p": float(p)}
        ax.set_title(f"{OUTCOMES[o]}\nKruskal-Wallis p = {p:.3g}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Viewing by composition class (CACNet)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "32_viewing_by_composition_class.png")

    agree = (df.cacnet_class == df.clip_class).mean()
    out = {"n": len(df),
           "spearman_rho": json.loads(rho.round(4).to_json()), "q": json.loads(q.round(5).to_json()),
           "significant_q<0.05": [{"predictor": p, "outcome": o, "rho": round(float(rho.loc[p, o]), 3), "q": float(q.loc[p, o])}
                                  for p in PREDICTORS for o in OUTCOMES if q.loc[p, o] < 0.05],
           "kruskal_by_cacnet_class": kw,
           "cacnet_class_counts": df.cacnet_class.value_counts().to_dict(),
           "clip_class_counts": df.clip_class.value_counts().to_dict(),
           "cacnet_clip_top_class_agreement": round(float(agree), 3)}
    with open(os.path.join(OUT, "results_painting_factors.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["significant_q<0.05"], indent=1))


if __name__ == "__main__":
    main()
