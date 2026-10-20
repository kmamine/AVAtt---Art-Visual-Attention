"""Intra- vs inter-image scanpath similarity: significance tests for every metric.

Usage (from the repository root):  python analysis/similarity_stats.py
  (after scanpath_similarity.py and scanpath_embeddings.py)
Writes figures 20, 21, 33, analysis/similarity/stats_table.csv, results_scanpath_similarity.json,
per_image_scanpath_similarity.csv.

For each metric (oriented so that higher = more similar):
  AUC        P(intra pair more similar than inter pair), over all pairs (0.5 = no difference)
  Wilcoxon   paired signed-rank test over paintings, on each painting's mean intra minus mean inter
  r_rb       matched-pairs rank-biserial correlation (effect size of that test, -1..1)
  q          Benjamini-Hochberg across all metrics
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analysis import INK, INK2, MUTED, OUT, SURFACE, plt, save
from paintfix import auc, bh
from scanpath_embeddings import MODELS
from scanpath_similarity import METRICS, SIM

EMBED_MEASURES = {"bertF1": (True, "patch-BERTScore F1"), "dtw": (False, "embedding DTW"),
                  "meanpool": (True, "mean-pooled cosine")}


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return np.nan
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    pairs = pd.read_csv(os.path.join(SIM, "pairs.csv"))
    df = pairs.merge(pd.read_csv(os.path.join(SIM, "scores_spatial.csv")), on="pair_id")
    meta = {m: {"group": "Spatial (eyefeatures)" if m != "FDISS" else "FDISS", "label": lab, "higher": hi}
            for m, (hi, lab) in METRICS.items()}
    for path in sorted(glob.glob(os.path.join(SIM, "scores_embed_*.csv"))):
        e = pd.read_csv(path)
        df = df.merge(e, on="pair_id")
        model = os.path.basename(path)[len("scores_embed_"):-4]
        for k, (hi, lab) in EMBED_MEASURES.items():
            meta[f"{model}__{k}"] = {"group": MODELS[model][0], "label": f"{MODELS[model][1].split(',')[0]} {lab}",
                                     "higher": hi, "model": model, "measure": k}

    rows, per_image = [], {}
    for m, info in meta.items():
        sign = 1 if info["higher"] else -1
        v = df[m] * sign
        intra, inter = v[df.kind == "intra"], v[df.kind == "inter"]
        img = df.assign(v=v).groupby(["image", "kind"]).v.mean().unstack()
        d = (img["intra"] - img["inter"]).dropna()
        per_image[m] = d * sign
        rows.append({"metric": m, **info, "AUC": auc(intra, inter), "cliffs_delta": 2 * auc(intra, inter) - 1,
                     "intra_mean": float(df.loc[df.kind == "intra", m].mean()),
                     "inter_mean": float(df.loc[df.kind == "inter", m].mean()),
                     "n_images": len(d), "images_intra_more_similar_pct": float(100 * (d > 0).mean()),
                     "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if len(d) > 5 else np.nan,
                     "r_rank_biserial": rank_biserial(d)})
    st = pd.DataFrame(rows)
    st["q"] = bh(st.wilcoxon_p.to_numpy())
    st.round(6).to_csv(os.path.join(SIM, "stats_table.csv"), index=False)
    pd.DataFrame(per_image).round(6).to_csv(os.path.join(OUT, "per_image_scanpath_similarity.csv"))

    # --- figure 20: spatial metrics, intra vs inter distributions
    spatial = [m for m in METRICS]
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5))
    for ax, m in zip(axes.ravel(), spatial):
        a, b = df.loc[df.kind == "intra", m].dropna(), df.loc[df.kind == "inter", m].dropna()
        bp = ax.boxplot([a, b], widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": INK2},
                        capprops={"color": INK2}, boxprops={"edgecolor": SURFACE, "lw": 1})
        for patch, c in zip(bp["boxes"], ["#2a78d6", "#eb6834"]):
            patch.set_facecolor(c)
        rr = st.set_index("metric").loc[m]
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Intra-image", "Inter-image"])
        ax.set_title(f"{meta[m]['label']} ({'higher' if meta[m]['higher'] else 'lower'} = similar)\n"
                     f"AUC {rr.AUC:.3f}, r = {rr.r_rank_biserial:.2f}, q = {rr.q:.1g}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Scanpath similarity (first 5 s): two observers of the same painting vs an observer and another "
                 "painting's scanpath mapped onto it", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "20_scanpath_similarity_distributions.png")

    # --- figure 21: discrimination of every metric, grouped
    group_colors = {"FDISS": "#2a78d6", "Spatial (eyefeatures)": "#86b6ef", "CNN (supervised)": "#eb6834",
                    "ViT (supervised)": "#eda100", "ViT (language-supervised)": "#e87ba4", "Self-supervised": "#1baf7a"}
    s = st.sort_values("AUC")
    fig, ax = plt.subplots(figsize=(10, max(6, 0.24 * len(s))))
    ax.barh(range(len(s)), s.AUC - 0.5, left=0.5, color=[group_colors[g] for g in s.group], height=0.75)
    for i, rrow in enumerate(s.itertuples()):
        star = "***" if rrow.q < 0.001 else "**" if rrow.q < 0.01 else "*" if rrow.q < 0.05 else ""
        ax.text(max(rrow.AUC, 0.5) + 0.004, i, f"{rrow.AUC:.3f}{star}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.label, fontsize=7.5)
    ax.set_xlim(0.45, s.AUC.max() + 0.06)
    ax.set_xlabel("AUC: P(intra-image pair more similar than inter-image pair)\n"
                  "* q<0.05, ** q<0.01, *** q<0.001 (paired Wilcoxon over paintings, BH)")
    ax.set_title("How well each metric separates real shared viewing from chance")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_colors.values()]
    ax.legend(handles, list(group_colors), loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "21_scanpath_metric_discrimination.png")

    # --- figure 33: image-encoder metrics, models x measures
    emb = st.dropna(subset=["model"]) if "model" in st else pd.DataFrame()
    if len(emb):
        grid = emb.pivot(index="model", columns="measure", values="AUC").reindex(list(MODELS)).dropna(how="all")
        qg = emb.pivot(index="model", columns="measure", values="q").reindex(grid.index)
        from matplotlib.colors import LinearSegmentedColormap
        seq = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#104281"])
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(grid) + 1.5))
        im = ax.imshow(grid[list(EMBED_MEASURES)].to_numpy(), cmap=seq, vmin=0.5, vmax=max(0.6, grid.max().max()), aspect="auto")
        for i in range(len(grid)):
            for j, k in enumerate(EMBED_MEASURES):
                v, q = grid.iloc[i][k], qg.iloc[i][k]
                star = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
                ax.text(j, i, f"{v:.3f}{star}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.5 + 0.6 * (grid.max().max() - 0.5) else INK)
        ax.set_xticks(range(len(EMBED_MEASURES)))
        ax.set_xticklabels([v[1].replace(" ", "\n", 1) for v in EMBED_MEASURES.values()], fontsize=9)
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels([f"{MODELS[m][1].split(',')[0]}  [{MODELS[m][0]}]" for m in grid.index], fontsize=8.5)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("AUC intra vs inter", color=INK2)
        cb.outline.set_visible(False)
        ax.set_title("Image-encoder scanpath similarity (96-px fixation patches)")
        fig.tight_layout()
        save(fig, "33_encoder_scanpath_similarity.png")

    out = {"design": {"intra_pairs": int((df.kind == "intra").sum()), "inter_pairs": int((df.kind == "inter").sum()),
                      "images": int(df.image.nunique()), "window": "first 5 s",
                      "inter_baseline": "observer of another painting (same exposure duration) mapped by painting coordinates"},
           "metrics": json.loads(st.drop(columns=["model", "measure"], errors="ignore").round(6).to_json(orient="records"))}
    with open(os.path.join(OUT, "results_scanpath_similarity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(st.sort_values("AUC", ascending=False)[["label", "group", "AUC", "r_rank_biserial", "images_intra_more_similar_pct", "q"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
