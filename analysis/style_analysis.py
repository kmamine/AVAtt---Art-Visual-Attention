"""Style (movement) analysis: attention, composition and objects per style, and which factor explains viewing best.

Usage (from the repository root):  python analysis/style_analysis.py
  (after painting_metadata.py, composition_*.py, faces_objects.py, features_baseline.py)
Writes figures 34-38, results_style.json and per_image_style.csv.

Viewing measures use only the first 5 s of every scanpath and fixations on the painting, so that paintings shown
for 5.5 s and 15 s are compared on the same window. Exposure duration is additionally included as a nuisance
covariate in every model. Movement groups with fewer than MIN_GROUP paintings are pooled as "Other" in the models
and omitted from per-style plots.
"""
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # small ridge fits are ~40x slower with threaded BLAS

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import DATA, INK, INK2, OUT, PX_PER_MM, SURFACE, plt, save
from paintfix import COMP, ci95, bh, load_all

ROOT = os.path.dirname(OUT)
WINDOW_S = 5.0
MIN_GROUP = 5
N_SPLITHALF = 10
N_REPEATS = 10
N_PERM = 200
SEED = 0
FAMILIES = ["thirds", "golden_grid", "centre", "diagonal", "golden_triangles", "dynamic_symmetry",
            "spiral_stretched", "spiral_true_ratio"]
FAM_LABEL = {"thirds": "Thirds", "golden_grid": "Golden grid", "centre": "Centre", "diagonal": "Diagonal",
             "golden_triangles": "Golden triangles", "dynamic_symmetry": "Dynamic symmetry",
             "spiral_stretched": "Spiral (stretched)", "spiral_true_ratio": "Spiral (true ratio)"}
OUTCOMES = {"entropy": "Saliency-map entropy (bits)", "dispersion_deg": "Fixation dispersion (°)",
            "agreement_cc": "Observer agreement (split-half CC)", "fix_dur_ms": "Median fixation duration (ms)",
            "sacc_amp_deg": "Median saccade amplitude (°)", "central_pct": "Fixations in central half (%)"}
OBJECTS = ["face", "person", "hand", "animal", "vessel / vehicle", "building", "tree / plant", "food / fruit", "sky", "water"]
FEATURES = {"edges": "Edges", "spatial_info": "Spatial info", "contrast": "Contrast", "colourfulness": "Colourfulness",
            "saturation": "Saturation", "nearness": "Nearness", "salient_object": "Salient object", "luminance": "Luminance"}
CMAP_DIV = None


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4b393", "#f0efec", "#86b6ef", "#2a78d6"])


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#104281"])


# ---------------------------------------------------------------- viewing measures on the first 5 s
def viewing_measures(det, fix, ppd, rng):
    rows = []
    for name, g in fix.groupby("image"):
        exposure = 15.0 if g.groupby("observer").t_s.max().median() > 8 else 5.5
        g = g[(g.t_s < WINDOW_S) & g.on_painting].sort_values(["observer", "onset_ms"])
        w, h = det[name]["crop_size"]
        px, py = g.u.to_numpy() * w, g.v.to_numpy() * h
        sc = 0.25
        ws, hs = max(int(w * sc), 1), max(int(h * sc), 1)
        m = np.zeros((hs, ws), np.float32)
        np.add.at(m, (np.clip((py * sc).astype(int), 0, hs - 1), np.clip((px * sc).astype(int), 0, ws - 1)), 1)
        m = cv2.GaussianBlur(m, (0, 0), ppd * sc)
        p = m / m.sum()
        ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        disp = float(np.hypot(px - px.mean(), py - py.mean()).mean() / ppd)
        dx = g.groupby("observer").u.diff().to_numpy() * w / PX_PER_MM
        dy = g.groupby("observer").v.diff().to_numpy() * h / PX_PER_MM
        amp = np.degrees(np.arctan(np.hypot(dx, dy) / g.distance_mm.fillna(g.distance_mm.mean()).to_numpy()))
        central = float(100 * (g.u.between(0.25, 0.75) & g.v.between(0.25, 0.75)).mean())
        obs = g.observer.unique()
        maps = {}
        for o, s in g.groupby("observer"):
            mm = np.zeros((hs, ws), np.float32)
            np.add.at(mm, (np.clip((s.v.to_numpy() * hs).astype(int), 0, hs - 1),
                           np.clip((s.u.to_numpy() * ws).astype(int), 0, ws - 1)), 1)
            maps[o] = mm
        ccs = []
        for _ in range(N_SPLITHALF):
            perm = rng.permutation(obs)
            a = cv2.GaussianBlur(sum(maps[o] for o in perm[: len(perm) // 2]), (0, 0), ppd * sc).ravel()
            b = cv2.GaussianBlur(sum(maps[o] for o in perm[len(perm) // 2:]), (0, 0), ppd * sc).ravel()
            ccs.append(np.corrcoef(a, b)[0, 1])
        rows.append({"image": name, "exposure_s": exposure, "entropy": ent, "dispersion_deg": disp,
                     "agreement_cc": float(np.mean(ccs)), "fix_dur_ms": float(g.duration_ms.median()),
                     "sacc_amp_deg": float(np.nanmedian(amp)), "central_pct": central})
    return pd.DataFrame(rows).set_index("image")


# ---------------------------------------------------------------- predictor blocks
def build_table(det, view):
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
    meta = meta.rename(columns={"attribution_confidence": "confidence"})
    fits = json.load(open(os.path.join(COMP, "fits.json")))
    pi = pd.read_csv(os.path.join(OUT, "per_image.csv"), index_col=0)
    fo = pd.read_csv(os.path.join(OUT, "per_image_faces_objects.csv"))
    feat = pd.read_csv(os.path.join(OUT, "per_image_feature_auc.csv"), index_col=0)
    ca = pd.read_csv(os.path.join(OUT, "per_image_composition_attention.csv"), index_col=0)

    df = view.join(meta[["movement_group", "artist", "genre", "confidence"]])
    for fam in FAMILIES:
        df[f"fit_{fam}"] = [fits[n][fam].get("score", np.nan) for n in df.index]
        df[f"enrich_{fam}"] = ca[f"{fam}__enrich_best"].reindex(df.index)
    for c in ["rule of thirds", "vertical", "horizontal", "diagonal", "curved", "triangle", "center", "symmetric", "pattern"]:
        df[f"cacnet_{c}"] = [det[n]["composition_probs"]["cacnet"][c] for n in df.index]
    df["cacnet_class"] = [max(det[n]["composition_probs"]["cacnet"], key=det[n]["composition_probs"]["cacnet"].get)
                          for n in df.index]
    area = fo.pivot_table(index="image", columns="category", values="area").reindex(df.index).fillna(0)
    enrich = fo.pivot_table(index="image", columns="category", values="log2_enrichment").reindex(df.index)
    for c in OBJECTS:
        df[f"area_{c}"] = area[c] if c in area else 0.0
        df[f"enrich_obj_{c}"] = enrich[c] if c in enrich else np.nan
    df["n_faces"] = [len(det[n]["faces"]) for n in df.index]
    df["n_subjects"] = [len(det[n]["subjects"]) for n in df.index]
    df["salient_area"] = [det[n]["salient_area_frac"] for n in df.index]
    df["has_horizon"] = [float(det[n]["horizon_v"] is not None) for n in df.index]
    df["colorfulness"] = pi.colorfulness.reindex(df.index)
    df["SI"] = pi.SI.reindex(df.index)
    df["mirror_symmetry"] = [det[n]["mirror_symmetry"] for n in df.index]
    for k in ["horizontal", "vertical", "diagonal"]:
        df[f"lines_{k}"] = [det[n]["line_orientation_share"][k] for n in df.index]
    df["aspect_ratio"] = [det[n]["aspect_ratio"] for n in df.index]
    df["painting_area"] = pi.painting_area_pct.reindex(df.index)
    for f in FEATURES:
        df[f"auc_{f}"] = feat[f].reindex(df.index)
    counts = df.movement_group.value_counts()
    df["style"] = df.movement_group.where(df.movement_group.map(counts) >= MIN_GROUP, "Other")
    return df


def blocks(df):
    style = pd.get_dummies(df["style"], prefix="style", dtype=float)
    return {
        "Style": style,
        "Composition": df[[f"fit_{f}" for f in FAMILIES] + [c for c in df if c.startswith("cacnet_") and c != "cacnet_class"]],
        "Content": df[["n_faces", "n_subjects", "salient_area", "has_horizon"] + [f"area_{c}" for c in OBJECTS]],
        "Low-level": df[["colorfulness", "SI", "mirror_symmetry", "lines_horizontal", "lines_vertical", "lines_diagonal"]],
        "Format": df[["aspect_ratio", "painting_area"]],
    }


# ---------------------------------------------------------------- cross-validated variance partitioning
def cv_r2(X, y, repeats=N_REPEATS, seed=SEED):
    scores = []
    for r in range(repeats):
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(X):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
            model.fit(X[tr], y[tr])
            pred[te] = model.predict(X[te])
        scores.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(scores)), float(np.std(scores))


def partition(df, outcome, rng):
    B = blocks(df)
    y = df[outcome].to_numpy(float)
    nuis = df[["exposure_s"]].to_numpy(float)
    ok = ~np.isnan(y)
    full = np.hstack([nuis] + [b.to_numpy(float) for b in B.values()])
    ok &= ~np.isnan(full).any(1)
    y, nuis, full = y[ok], nuis[ok], full[ok]
    mats = {k: b.to_numpy(float)[ok] for k, b in B.items()}
    base, _ = cv_r2(nuis, y)
    r_full, sd_full = cv_r2(full, y)
    out = {"n": int(ok.sum()), "R2_nuisance": base, "R2_full": r_full, "R2_full_sd": sd_full, "blocks": {}}
    for k, X in mats.items():
        alone, _ = cv_r2(np.hstack([nuis, X]), y)
        rest = np.hstack([nuis] + [m for kk, m in mats.items() if kk != k])
        without, _ = cv_r2(rest, y)
        unique = r_full - without
        # permutation test of the unique contribution: shuffle the block's rows
        null = []
        for _ in range(N_PERM):
            Xp = X[rng.permutation(len(X))]
            r_perm, _ = cv_r2(np.hstack([rest, Xp]), y, repeats=2)
            null.append(r_perm - without)
        p = (1 + np.sum(np.array(null) >= unique)) / (1 + N_PERM)
        out["blocks"][k] = {"alone_minus_nuisance": alone - base, "unique": unique, "p_unique": float(p)}
    return out


# ---------------------------------------------------------------- figures
def heat(ax, M, rowlabels, collabels, cmap, vmin, vmax, fmt="{:.2f}", stars=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            s = fmt.format(v) + (stars[i][j] if stars is not None else "")
            ax.text(j, i, s, ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(collabels)))
    ax.set_xticklabels(collabels, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rowlabels)))
    ax.set_yticklabels(rowlabels, fontsize=8.5)
    ax.grid(False)
    return im


def main():
    rng = np.random.default_rng(SEED)
    det, fix, ppd = load_all()
    view = viewing_measures(det, fix, ppd, rng)
    df = build_table(det, view)
    df.round(5).to_csv(os.path.join(OUT, "per_image_style.csv"))
    styles = [s for s in df["style"].value_counts().index if s != "Other"]
    n_style = df["style"].value_counts()
    res = {"window_s": WINDOW_S, "min_group": MIN_GROUP, "styles": {s: int(n_style[s]) for s in styles}}

    # residualise outcomes on exposure for display (style effects net of exposure)
    def adj(col):
        sub = df[[col, "exposure_s"]].dropna()
        slope = np.polyfit(sub.exposure_s, sub[col], 1)[0]
        return df[col] - slope * (df.exposure_s - df.exposure_s.mean())

    # ------------------------------------------------ fig 34: attention by style
    order = df[df["style"].isin(styles)].groupby("style").agreement_cc.median().sort_values().index
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    kw = {}
    for ax, (o, lab) in zip(axes.ravel(), OUTCOMES.items()):
        v = adj(o)
        groups = [v[df["style"] == s].dropna() for s in order]
        H, p = stats.kruskal(*groups)
        kw[o] = p
        ax.boxplot(groups, vert=False, labels=[f"{s} ({n_style[s]})" for s in order], showfliers=False,
                   patch_artist=True, medianprops={"color": INK}, boxprops={"facecolor": "#86b6ef", "edgecolor": SURFACE})
        ax.set_title(f"{lab}\nKruskal-Wallis p = {p:.2g}", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
        if ax not in axes[:, 0]:
            ax.set_yticklabels([])
    q = bh(np.array(list(kw.values())))
    res["attention_kruskal"] = {o: {"p": float(p), "q": float(qq)} for (o, p), qq in zip(kw.items(), q)}
    fig.suptitle("Attention by style (first 5 s, adjusted for exposure duration)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "34_attention_by_style.png")

    # ------------------------------------------------ fig 35: composition by style
    fitz = df[[f"fit_{f}" for f in FAMILIES]]
    fitz = (fitz - fitz.mean()) / fitz.std()
    Mfit = np.array([[fitz.loc[df["style"] == s, f"fit_{f}"].mean() for f in FAMILIES] for s in order])
    pfit = [stats.kruskal(*[df.loc[df["style"] == s, f"fit_{f}"].dropna() for s in order]).pvalue for f in FAMILIES]
    qfit = bh(np.array(pfit))
    Menr = np.array([[2 ** df.loc[df["style"] == s, f"enrich_{f}"].mean() for f in FAMILIES] for s in order])
    rho_by_style = {}
    for s in order:
        sub = df[df["style"] == s]
        rr = []
        for f in FAMILIES:
            d = sub[[f"fit_{f}", f"enrich_{f}"]].dropna()
            rr.append(stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 8 else np.nan)
        rho_by_style[s] = rr
    Mrho = np.array([rho_by_style[s] for s in order])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    im = heat(axes[0], Mfit, list(order), [f"{FAM_LABEL[f]}{'*' if qq < .05 else ''}" for f, qq in zip(FAMILIES, qfit)],
              div_cmap(), -1, 1)
    axes[0].set_title("Composition fit (z-score, image only)\n* family differs across styles (BH q<0.05)", fontsize=10)
    plt.colorbar(im, ax=axes[0], shrink=0.7).outline.set_visible(False)
    im = heat(axes[1], Menr, list(order), [FAM_LABEL[f] for f in FAMILIES], div_cmap(), 0.6, 1.4)
    axes[1].set_title("Fixations on the template vs baseline (ratio)", fontsize=10)
    axes[1].set_yticklabels([])
    plt.colorbar(im, ax=axes[1], shrink=0.7).outline.set_visible(False)
    im = heat(axes[2], Mrho, list(order), [FAM_LABEL[f] for f in FAMILIES], div_cmap(), -0.6, 0.6)
    axes[2].set_title("Within style: Spearman ρ(fit, fixation enrichment)\n(styles with ≥8 paintings)", fontsize=10)
    axes[2].set_yticklabels([])
    plt.colorbar(im, ax=axes[2], shrink=0.7).outline.set_visible(False)
    fig.tight_layout()
    save(fig, "35_composition_by_style.png")
    res["composition_fit_kruskal"] = {f: {"p": float(p), "q": float(qq)} for f, p, qq in zip(FAMILIES, pfit, qfit)}
    res["composition_fit_by_style_z"] = {s: {f: round(float(v), 3) for f, v in zip(FAMILIES, row)} for s, row in zip(order, Mfit)}
    ct = pd.crosstab(df.loc[df["style"].isin(styles), "style"], df.loc[df["style"].isin(styles), "cacnet_class"])
    chi = stats.chi2_contingency(ct)
    res["cacnet_class_by_style"] = {"chi2": float(chi[0]), "p": float(chi[1]), "table": ct.to_dict(orient="index")}

    # ------------------------------------------------ fig 36: objects by style
    Mpres = np.array([[100 * (df.loc[df["style"] == s, f"area_{c}"] > 0).mean() for c in OBJECTS] for s in order])
    Menr_o = np.full((len(order), len(OBJECTS)), np.nan)
    for i, s in enumerate(order):
        for j, c in enumerate(OBJECTS):
            v = df.loc[(df["style"] == s) & (df[f"area_{c}"] > 0), f"enrich_obj_{c}"].dropna()
            if len(v) >= 3:
                Menr_o[i, j] = 2 ** v.mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    im = heat(axes[0], Mpres, list(order), OBJECTS, seq_cmap(), 0, 100, fmt="{:.0f}")
    axes[0].set_title("Paintings containing the category (%)", fontsize=10)
    plt.colorbar(im, ax=axes[0], shrink=0.7).outline.set_visible(False)
    im = heat(axes[1], Menr_o, list(order), OBJECTS, div_cmap(), 0, 4, fmt="{:.1f}")
    axes[1].set_title("Fixation enrichment over baseline (×), where present in ≥3 paintings", fontsize=10)
    axes[1].set_yticklabels([])
    plt.colorbar(im, ax=axes[1], shrink=0.7).outline.set_visible(False)
    fig.tight_layout()
    save(fig, "36_objects_by_style.png")
    res["objects_presence_pct"] = {s: {c: round(float(v), 1) for c, v in zip(OBJECTS, row)} for s, row in zip(order, Mpres)}
    res["objects_enrichment"] = {s: {c: (None if np.isnan(v) else round(float(v), 2)) for c, v in zip(OBJECTS, row)}
                                 for s, row in zip(order, Menr_o)}
    face_by_style = {s: df.loc[(df["style"] == s) & (df["area_face"] > 0), "enrich_obj_face"].dropna() for s in order}
    face_by_style = {s: v for s, v in face_by_style.items() if len(v) >= 3}
    if len(face_by_style) > 1:
        res["face_enrichment_kruskal_p"] = float(stats.kruskal(*face_by_style.values()).pvalue)

    # ------------------------------------------------ fig 37: image features by style
    Mf = np.array([[df.loc[df["style"] == s, f"auc_{f}"].mean() for f in FEATURES] for s in order])
    pf = [stats.kruskal(*[df.loc[df["style"] == s, f"auc_{f}"].dropna() for s in order]).pvalue for f in FEATURES]
    qf = bh(np.array(pf))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = heat(ax, Mf, list(order), [f"{FEATURES[f]}{'*' if qq < .05 else ''}" for f, qq in zip(FEATURES, qf)],
              div_cmap(), 0.35, 0.65, fmt="{:.2f}")
    ax.set_title("Feature AUC (fixated vs baseline) by style\n* differs across styles (BH q<0.05)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7).outline.set_visible(False)
    save(fig, "37_features_by_style.png")
    res["features_by_style"] = {s: {f: round(float(v), 3) for f, v in zip(FEATURES, row)} for s, row in zip(order, Mf)}
    res["features_kruskal"] = {f: {"p": float(p), "q": float(qq)} for f, p, qq in zip(FEATURES, pf, qf)}

    # ------------------------------------------------ fig 38: what explains viewing
    part = {o: partition(df, o, rng) for o in OUTCOMES}
    res["variance_partitioning"] = part
    names = list(blocks(df))
    colors = {"Style": "#2a78d6", "Composition": "#eb6834", "Content": "#1baf7a", "Low-level": "#eda100", "Format": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(OUTCOMES))
    wd = 0.16
    for k, nm in enumerate(names):
        a = [part[o]["blocks"][nm]["alone_minus_nuisance"] for o in OUTCOMES]
        u = [part[o]["blocks"][nm]["unique"] for o in OUTCOMES]
        axes[0].bar(x + (k - 2) * wd, a, wd, color=colors[nm], label=nm)
        axes[1].bar(x + (k - 2) * wd, u, wd, color=colors[nm], label=nm)
        for i, o in enumerate(OUTCOMES):
            if part[o]["blocks"][nm]["p_unique"] < 0.05 and u[i] > 0:
                axes[1].text(x[i] + (k - 2) * wd, max(u[i], 0) + 0.004, "*", ha="center", fontsize=10, color=INK)
    for ax, t in zip(axes, ["Variance explained by each block alone\n(cross-validated R², beyond exposure duration)",
                            "Unique contribution of each block\n(R² lost when the block is removed; * permutation p<0.05)"]):
        ax.set_xticks(x)
        ax.set_xticklabels([OUTCOMES[o].split(" (")[0] for o in OUTCOMES], rotation=20, ha="right", fontsize=9)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylabel("Cross-validated R²")
        ax.set_title(t, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "38_style_vs_composition_variance.png")

    with open(os.path.join(OUT, "results_style.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps({"attention_kruskal": res["attention_kruskal"], "composition_fit_kruskal": res["composition_fit_kruskal"],
                      "cacnet_chi2_p": res["cacnet_class_by_style"]["p"],
                      "partition": {o: {"full": round(v["R2_full"], 3), **{k: (round(b["alone_minus_nuisance"], 3), round(b["unique"], 3), b["p_unique"])
                                                                         for k, b in v["blocks"].items()}} for o, v in part.items()}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
