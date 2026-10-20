"""Build every number, table and figure used by the AVAtt preprint from the analysis outputs.

Usage (from the repository root, after scripts/run_all.py):  python paper/make_assets.py
Writes paper/generated/macros.tex, paper/generated/tab_*.tex and paper/figures/*.
Re-run after any analysis update; main.tex only reads these files.
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AN = os.environ.get("AVATT_RESULTS", os.path.join(ROOT, "results"))
DATA = os.environ.get("AVATT_DATA", os.path.join(ROOT, "data", "AVAtt"))
GEN = os.path.join(HERE, "generated")
FIG = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from analysis import INK, INK2, MUTED, SURFACE, plt, save as _save  # noqa: E402

os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def J(name):
    with open(os.path.join(AN, name)) as f:
        return json.load(f)


def fmt(v, d=2):
    return f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight")
    plt.close(fig)


macros = {}


def M(name, value):
    macros[name] = value


# ------------------------------------------------------------------ participants (questionnaire data)
AGES = {18: 27, 19: 28, 20: 35, 23: 23, 25: 5, 26: 5, 27: 3, 30: 6, 33: 7, 36: 4, 40: 2, 43: 2, 49: 2, 56: 2, 62: 1}
GENDER = {"Male": 110, "Female": 42}
EXPERTISE = {"Novice": 146, "Amateur": 5, "Expert": 1}
COUNTRIES = {"France": 130, "Algeria": 5, "Morocco": 4, "Tunisia": 3, "China": 2, "Benin": 1, "Cameroon": 1,
             "Guinea": 1, "Italy": 1, "Lebanon": 1, "Peru": 1, "Syria": 1, "Vietnam": 1}


def participants():
    ages = np.repeat(list(AGES), list(AGES.values()))
    n = int(sum(GENDER.values()))
    M("nParticipants", n)
    M("ageMean", f"{ages.mean():.1f}")
    M("ageSD", f"{ages.std(ddof=1):.1f}")
    M("ageMedian", f"{np.median(ages):.0f}")
    M("ageMin", f"{ages.min()}")
    M("ageMax", f"{ages.max()}")
    M("pctFemale", f"{100 * GENDER['Female'] / n:.1f}")
    M("pctMale", f"{100 * GENDER['Male'] / n:.1f}")
    M("nFemale", GENDER["Female"])
    M("nMale", GENDER["Male"])
    M("pctNovice", f"{100 * EXPERTISE['Novice'] / n:.1f}")
    M("nAmateur", EXPERTISE["Amateur"])
    M("nExpert", EXPERTISE["Expert"])
    M("pctFrance", f"{100 * COUNTRIES['France'] / sum(COUNTRIES.values()):.1f}")
    M("nCountries", len(COUNTRIES))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw={"width_ratios": [2.2, 0.9, 1.0, 2.2]})
    ax = axes[0]
    ax.bar([str(a) for a in AGES], list(AGES.values()), color="#2a78d6", edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Participants")
    ax.set_title(f"Age (mean {ages.mean():.1f} ± {ages.std(ddof=1):.1f})")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    for ax, d, title in [(axes[1], GENDER, "Gender"), (axes[2], EXPERTISE, "Art expertise")]:
        ax.bar(list(d), list(d.values()), color=["#2a78d6", "#eb6834", "#1baf7a"][:len(d)], edgecolor=SURFACE, linewidth=1.5)
        for i, v in enumerate(d.values()):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    ax = axes[3]
    c = dict(sorted(COUNTRIES.items(), key=lambda t: -t[1]))
    ax.barh(list(c)[::-1], list(c.values())[::-1], color="#1baf7a", edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Participants (log scale)")
    ax.set_title("Country of origin")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "participants.png")


# ------------------------------------------------------------------ dataset counts and recordings
def dataset():
    r = J("results.json")
    a = r["all"]
    s = r["setup"]
    M("nImages", a["images"])
    M("nScanpaths", fmt(a["scanpaths"]))
    M("nFixations", fmt(a["fixations"]))
    M("spPerImageMin", a["scanpaths_per_image"]["min"])
    M("spPerImageMedian", f"{a['scanpaths_per_image']['median']:.0f}")
    M("spPerImageMax", a["scanpaths_per_image"]["max"])
    M("viewTimeMean", f"{a['viewing_time_s_mean']:.1f}")
    M("fixPerSp", f"{a['fixations_per_scanpath_mean']:.1f}")
    M("fixDurMean", f"{a['fixation_duration_ms']['mean']:.0f}")
    M("fixDurMedian", f"{a['fixation_duration_ms']['median']:.0f}")
    M("fixDurSD", f"{a['fixation_duration_ms']['sd']:.0f}")
    M("saccAmpMean", f"{a['saccade_amplitude_deg']['mean']:.2f}")
    M("saccAmpMedian", f"{a['saccade_amplitude_deg']['median']:.2f}")
    M("saccAmpSD", f"{a['saccade_amplitude_deg']['sd']:.2f}")
    M("pctHorizontal", f"{a['horizontal_saccades_pct']:.1f}")
    M("distMean", f"{a['viewing_distance_mm']['mean'] / 10:.1f}")
    M("distSD", f"{a['viewing_distance_mm']['sd'] / 10:.1f}")
    M("distPctRange", f"{a['viewing_distance_mm']['pct_500_650']:.1f}")
    M("pctCentralHalf", f"{a['fixations_in_central_half_pct']:.1f}")
    M("meanFixU", f"{a['mean_fixation_position_norm'][0]:.3f}")
    M("meanFixV", f"{a['mean_fixation_position_norm'][1]:.3f}")
    M("pctOnPainting", f"{a['fixations_on_painting_pct']:.1f}")
    M("paintingAreaPct", f"{a['painting_area_pct_mean']:.1f}")
    M("imgColourMean", f"{a['image_colorfulness']['mean']:.1f}")
    M("imgColourSD", f"{a['image_colorfulness']['sd']:.1f}")
    M("imgSIMean", f"{a['image_SI']['mean']:.1f}")
    M("imgSISD", f"{a['image_SI']['sd']:.1f}")
    M("patchColourMean", f"{a['fixation_patch_colorfulness_mean']:.1f}")
    M("patchSIMean", f"{a['fixation_patch_SI_mean']:.1f}")
    M("iocMean", f"{a['IOC']['mean']:.2f}")
    M("iocSD", f"{a['IOC']['sd']:.2f}")
    M("iocMin", f"{a['IOC']['min']:.2f}")
    M("iocMax", f"{a['IOC']['max']:.2f}")
    M("screenWidthMM", f"{s['screen_width_mm']:.0f}")
    M("pxPerMM", f"{s['px_per_mm']:.3f}")
    M("pxPerDeg", f"{s['px_per_degree']:.2f}")
    cs = r["image_colorfulness_vs_SI"]
    M("colSIpearson", f"{cs['pearson_r']:.2f}")
    M("colSIspearman", f"{cs['spearman_rho']:.2f}")

    # comparison with public free-viewing datasets (only well-established figures; '--' where not applicable)
    rows = [
        r"MIT1003~\cite{judd2009mit} & Natural & 1{,}003 & eye & \checkmark & \checkmark & -- \\",
        r"CAT2000~\cite{borji2015cat2000} & 20 categories & 4{,}000 & eye & \checkmark & \checkmark & -- \\",
        r"SALICON~\cite{jiang2015salicon} & Natural & 20{,}000 & mouse & \checkmark & \checkmark & -- \\",
        r"Le Meur \etal~\cite{lemeur2020paintings} & Paintings (5 styles) & 150 & eye & -- & \checkmark & -- \\",
        rf"\textbf{{AVAtt (ours)}} & Paintings (16 movements) & {a['images']} & eye & \checkmark & \checkmark & \checkmark \\",
    ]
    with open(os.path.join(GEN, "tab_compare.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


# ------------------------------------------------------------------ pooled figures (no breakdown by session)
def pooled_figures():
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    dur, amp, dist, n_obs = [], [], [], []
    for fn in os.listdir(os.path.join(DATA, "scanpaths")):
        with open(os.path.join(DATA, "scanpaths", fn)) as f:
            sps = [json.loads(l)["fixations"] for l in f]
        n_obs.append(len(sps))
        for fx in sps:
            d = np.array([p["distance_mm"] if p["distance_mm"] is not None else np.nan for p in fx], float)
            x = np.array([p["x"] for p in fx], float)
            y = np.array([p["y"] for p in fx], float)
            dur += [p["duration_ms"] for p in fx]
            dist += list(d[~np.isnan(d)] / 10)
            dmm = np.where(np.isnan(d), np.nanmean(d) if np.isfinite(np.nanmean(d)) else 568.0, d)[1:]
            amp += list(np.degrees(np.arctan(np.hypot(np.diff(x), np.diff(y)) / 3.6161 / dmm)))

    def hist(ax, v, bins, color, xlabel, title):
        ax.hist(v, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(np.median(v), color=INK, lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title} (median {np.median(v):.{0 if np.median(v) > 20 else 1}f})")
        ax.grid(axis="x", visible=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    hist(axes[0], dur, np.arange(0, 2001, 25), BLUE, "Fixation duration (ms)", "Fixation duration")
    hist(axes[1], amp, np.arange(0, 30.5, 0.5), ORANGE, "Saccade amplitude (°)", "Saccade amplitude")
    hist(axes[2], dist, np.arange(40, 80.5, 0.5), "#1baf7a", "Eye-screen distance (cm)", "Viewing distance")
    axes[2].axvspan(50, 65, color="#e1e0d9", alpha=0.6, lw=0, zorder=0)
    vals, counts = np.unique(n_obs, return_counts=True)
    axes[3].bar([str(v) for v in vals], counts, color="#6250d6", edgecolor=SURFACE)
    axes[3].set_xlabel("Observers per painting")
    axes[3].set_title(f"Observers per painting (median {np.median(n_obs):.0f})")
    axes[3].grid(axis="x", visible=False)
    for ax in axes[:3]:
        ax.set_ylabel("Fixations" if ax is not axes[1] else "Saccades")
    axes[3].set_ylabel("Paintings")
    fig.tight_layout()
    save(fig, "eye_movement_stats.png")

    pi = pd.read_csv(os.path.join(AN, "per_image.csv"), index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    axes[0].hist(pi.colorfulness, bins=25, color=BLUE, edgecolor=SURFACE)
    axes[0].set_xlabel("Colourfulness")
    axes[0].set_ylabel("Paintings")
    axes[0].set_title("Colourfulness")
    axes[1].hist(pi.SI, bins=25, color=ORANGE, edgecolor=SURFACE)
    axes[1].set_xlabel("Spatial information")
    axes[1].set_title("Spatial information")
    axes[2].scatter(pi.SI, pi.colorfulness, s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    axes[2].set_xlabel("Spatial information")
    axes[2].set_ylabel("Colourfulness")
    axes[2].set_title("Joint distribution")
    for ax in axes[:2]:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "image_stats.png")

    fig, ax = plt.subplots(figsize=(5, 3.3))
    ax.hist(pi.IOC, bins=np.arange(0.15, 1.001, 0.025), color=BLUE, edgecolor=SURFACE)
    ax.axvline(pi.IOC.mean(), color=INK, lw=1, ls="--")
    ax.set_xlabel("Inter-observer congruency")
    ax.set_ylabel("Paintings")
    ax.set_title(f"IOC per painting (mean {pi.IOC.mean():.2f})")
    ax.grid(axis="x", visible=False)
    save(fig, "ioc.png")


# ------------------------------------------------------------------ temporal
def temporal():
    t = J("results_temporal.json")
    r = t["time_to_reach_CC"]
    M("ccNine", f"{r['CC_0.9']:.2f}")
    M("ccEight", f"{r['CC_0.8']:.2f}")
    nss = t["NSS_loo_by_window"]
    M("nssFirstSec", f"{nss['0.50']:.2f}")
    M("nssLastSec", f"{list(nss.values())[-1]:.2f}")


# ------------------------------------------------------------------ benchmark
LABEL_ORDER = ["Human (other half)", "Single observer", "DeepGaze IIE", "TranSalNet-Res", "UNISAL", "GBVS",
               "Itti-Koch", "Spectral residual", "Salient objects (BiRefNet)", "Centre bias (dataset)",
               "Centre bias (Gaussian)"]


def benchmark():
    b = J("results_benchmark.json")
    bm = b["benchmark"]
    metrics = ["AUC-Judd", "sAUC", "NSS", "CC", "SIM", "KL"]
    best = {m: (min if m == "KL" else max)(v[m]["mean"] for k, v in bm.items() if k not in ("Human (other half)",)) for m in metrics}
    rows = []
    for k in LABEL_ORDER:
        if k not in bm:
            continue
        cells = []
        for m in metrics:
            v = bm[k][m]["mean"]
            s = f"{v:.3f}"
            if k != "Human (other half)" and v == best[m]:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rows.append(f"{k} & " + " & ".join(cells) + " \\\\")
        if k in ("Single observer", "Salient objects (BiRefNet)"):
            rows.append("\\midrule")
    with open(os.path.join(GEN, "tab_benchmark.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    h = bm["Human (other half)"]
    M("humanAUC", f"{h['AUC-Judd']['mean']:.3f}")
    M("humanSAUC", f"{h['sAUC']['mean']:.3f}")
    M("humanNSS", f"{h['NSS']['mean']:.2f}")
    M("humanCC", f"{h['CC']['mean']:.3f}")
    cb = bm["Centre bias (dataset)"]
    M("centreAUC", f"{cb['AUC-Judd']['mean']:.3f}")
    M("centreNSS", f"{cb['NSS']['mean']:.2f}")
    M("centreCC", f"{cb['CC']['mean']:.3f}")
    models = [k for k in bm if k in LABEL_ORDER[2:9]]
    if models:
        top = max(models, key=lambda k: bm[k]["NSS"]["mean"])
        M("bestModel", top)
        M("bestModelCC", f"{bm[top]['CC']['mean']:.3f}")
        M("bestModelNSS", f"{bm[top]['NSS']['mean']:.2f}")
        M("bestModelAUC", f"{bm[top]['AUC-Judd']['mean']:.3f}")
        M("bestModelSAUC", f"{bm[top]['sAUC']['mean']:.3f}")
    conv = b["observer_convergence"]
    M("kNinetyFiveAUC", conv["AUC-Judd"]["k_reaching_95pct_of_max"])
    M("kNinetyFiveCC", conv["CC"]["k_reaching_95pct_of_max"])
    M("convImages", b["setup"]["convergence_images"])
    M("benchOnPainting", fmt(b["setup"].get("fixations_on_paintings", 0)))


# ------------------------------------------------------------------ scanpath similarity
def similarity():
    st = pd.read_csv(os.path.join(AN, "similarity", "stats_table.csv"))
    st = st.sort_values("AUC", ascending=False)
    spatial = st[st.group.isin(["FDISS", "Spatial (eyefeatures)"])]
    rows = []
    for r in spatial.itertuples():
        q = "$<10^{-3}$" if r.q < 1e-3 else f"{r.q:.3f}"
        rows.append(f"{r.label.replace('é', '\\\'e')} & {'$\\uparrow$' if r.higher else '$\\downarrow$'} & {r.intra_mean:.4g} & "
                    f"{r.inter_mean:.4g} & {r.AUC:.3f} & {r.r_rank_biserial:.2f} & {r.images_intra_more_similar_pct:.1f} & {q} \\\\")
    with open(os.path.join(GEN, "tab_scanpath_spatial.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    emb = st[~st.group.isin(["FDISS", "Spatial (eyefeatures)"])].copy()
    if len(emb):
        emb["model"] = emb.metric.str.split("__").str[0]
        emb["measure"] = emb.metric.str.split("__").str[1]
        piv = emb.pivot(index="model", columns="measure", values="AUC")
        grp = emb.groupby("model").group.first()
        lab = emb.groupby("model").label.first().str.replace(r" (patch-BERTScore F1|embedding DTW|mean-pooled cosine)$", "", regex=True)
        order = ["vgg16", "resnet50", "convnext_tiny", "vit_b16_in21k", "clip_vit_b16", "mae_vit_b16",
                 "dinov2_vit_b14", "dinov3_vit_b16", "ijepa_vit_h14", "vicreg_resnet50"]
        bestcol = {c: piv[c].max() for c in piv}
        rows = []
        for m in [o for o in order if o in piv.index]:
            cells = [f"\\textbf{{{piv.loc[m, c]:.3f}}}" if piv.loc[m, c] == bestcol[c] else f"{piv.loc[m, c]:.3f}"
                     for c in ["bertF1", "dtw", "meanpool"]]
            rows.append(f"{lab[m]} & {grp[m]} & " + " & ".join(cells) + " \\\\")
        with open(os.path.join(GEN, "tab_scanpath_encoders.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        top = emb.iloc[0]
        M("bestEncoderLabel", top.label)
        M("bestEncoderAUC", f"{top.AUC:.3f}")
    fd = st[st.metric == "FDISS"].iloc[0]
    M("fdissAUC", f"{fd.AUC:.3f}")
    M("fdissPct", f"{fd.images_intra_more_similar_pct:.1f}")
    M("fdissR", f"{fd.r_rank_biserial:.2f}")
    pairs = pd.read_csv(os.path.join(AN, "similarity", "pairs.csv"))
    M("nIntraPairs", fmt(int((pairs.kind == "intra").sum())))
    mmd = st[st.metric == "MM_duration"].iloc[0]
    M("mmDurationQ", f"{mmd.q:.2f}")


# ------------------------------------------------------------------ features, faces, composition, factors
def content():
    fe = J("results_features.json")["features"]
    for k, name in [("edges", "Edges"), ("spatial_info", "SI"), ("contrast", "Contrast"), ("colourfulness", "Colour"),
                    ("nearness", "Near"), ("saturation", "Sat"), ("salient_object", "SOD"), ("luminance", "Lum"),
                    ("centre_distance", "Centre")]:
        M(f"auc{name}", f"{fe[k]['mean_auc']:.3f}")
    fo = J("results_faces_objects.json")
    f = fo["faces"]
    M("nFaces", f["n_faces"])
    M("nPaintingsFaces", f["paintings_with_faces"])
    M("faceObsPct", f"{100 * f['obs_share']:.1f}")
    M("faceAreaPct", f"{100 * f['area_share']:.1f}")
    M("faceBasePct", f"{100 * f['base_share']:.1f}")
    M("faceLatency", f"{f['latency_obs_s']:.2f}")
    M("faceLatencyBase", f"{f['latency_base_s']:.2f}")
    cats = {c["category"]: c for c in fo["categories"]}
    M("faceReachOne", f"{100 * cats['face']['reach1_obs']:.0f}")
    M("faceReachOneBase", f"{100 * cats['face']['reach1_base']:.0f}")
    M("faceEnrich", f"{2 ** cats['face']['enrich']:.1f}")
    M("skyEnrich", f"{2 ** cats['sky']['enrich']:.2f}")
    rows = []
    for c in sorted(fo["categories"], key=lambda c: -c["enrich"]):
        q = "$<10^{-3}$" if c["q"] < 1e-3 else f"{c['q']:.3f}"
        rows.append(f"{c['category']} & {c['n_paintings']} & {100 * c['area_share']:.1f} & {100 * c['obs_share']:.1f} & "
                    f"{100 * c['base_share']:.1f} & {2 ** c['enrich']:.2f} & {100 * c['reach1_obs']:.0f} / {100 * c['reach1_base']:.0f} & {q} \\\\")
    with open(os.path.join(GEN, "tab_objects.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))

    co = J("results_composition.json")
    fs = co["focal_subjects"]
    M("subjObsPct", f"{100 * fs['obs_share']:.1f}")
    M("subjBasePct", f"{100 * fs['base_share']:.1f}")
    M("subjAreaPct", f"{100 * fs['area_share']:.1f}")
    lat = {x["key"]: x for x in co["first_fixation_latency"]}
    M("subjLatencyGain", f"{-lat['subject']['diff_mean_s']:.2f}")
    M("spiralEyeGain", f"{-lat['spiral_true_ratio']['diff_mean_s']:.2f}")
    M("spiralEyeP", f"{lat['spiral_true_ratio']['p']:.3f}")
    so = co["salient_object_map_nss"]
    M("sodNSS", f"{so['obs']:.2f}")
    M("sodNSSbase", f"{so['base']:.2f}")
    lf = co["line_following"]
    M("lineFollowP", f"{lf['p']:.3f}")
    M("spiralFollowP", f"{co['spiral_following_follow']['p']:.3f}")
    rows = []
    labels = {"thirds": "Rule of thirds", "golden_grid": "Golden grid", "centre": "Centre", "diagonal": "Diagonal",
              "golden_triangles": "Golden triangles", "dynamic_symmetry": "Dynamic symmetry",
              "spiral_stretched": "Golden spiral (stretched)", "spiral_true_ratio": "Golden spiral (true ratio)"}
    nvar = {"thirds": 1, "golden_grid": 1, "centre": 1, "diagonal": 2, "golden_triangles": 2, "dynamic_symmetry": 1,
            "spiral_stretched": 8, "spiral_true_ratio": 8}
    fits = json.load(open(os.path.join(AN, "composition", "fits.json")))
    for x in co["composition_elements"]:
        fam = x["family"]
        n05 = sum((fits[n][fam].get("p") or 1) < 0.05 for n in fits)
        nq = sum(fits[n][fam].get("follows", False) for n in fits)
        rho = x["rho_fit_vs_enrichment"]
        prho = "$<10^{-3}$" if x["p_rho"] < 1e-3 else f"{x['p_rho']:.3f}"
        rows.append(f"{labels[fam]} & {nvar[fam]} & {n05} & {nq} & {2 ** x['follow_mean']:.2f} & {2 ** x['rest_mean']:.2f} & "
                    f"{rho:.2f} & {prho} \\\\")
    with open(os.path.join(GEN, "tab_composition.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows))

    pf = J("results_painting_factors.json")
    M("cacClipAgree", f"{100 * pf['cacnet_clip_top_class_agreement']:.0f}")
    M("kwEntropyP", f"{pf['kruskal_by_cacnet_class']['entropy']['p']:.1g}")
    sig = {(s["predictor"], s["outcome"]): s for s in pf["significant_q<0.05"]}
    rho = pf["spearman_rho"]
    M("rhoFacesEntropy", f"{rho['entropy']['n_faces']:.2f}")
    M("rhoFacesDur", f"{rho['fix_dur_ms']['n_faces']:.2f}")
    M("rhoAspectDisp", f"{rho['dispersion_deg']['aspect_ratio']:.2f}")
    M("rhoHorizonEntropy", f"{rho['entropy']['has_horizon']:.2f}")
    M("rhoDiagCC", f"{rho['human_CC']['lines_diagonal']:.2f}")


# ------------------------------------------------------------------ figures
COPY = {"06_polar_absolute_direction.png": "polar_abs.png", "07_polar_relative_direction.png": "polar_rel.png",
        
        "08_central_bias.png": "central_bias.png",
        
        "11_hsv_fixations_vs_images.png": "hsv.png", 
        "15_benchmark_baselines.png": "benchmark.png",
        "16_observer_convergence.png": "convergence.png", "17_duration_amplitude_over_time.png": "temporal_dur_amp.png",
        "18_agreement_over_time.png": "temporal_agreement.png", "19_map_convergence.png": "temporal_convergence.png",
        "20_scanpath_similarity_distributions.png": "sp_spatial.png", "21_scanpath_metric_discrimination.png": "sp_discrimination.png",
        "33_encoder_scanpath_similarity.png": "sp_encoders.png", "22_composition_templates.png": "templates.png",
        "23_composition_contact_sheet.png": "contact_sheet.png", "24_composition_element_enrichment.png": "comp_enrichment.png",
        "25_subjects_composition_over_time.png": "comp_time.png", "26_first_fixation_latency.png": "comp_latency.png",
        "27_line_following.png": "line_following.png", "28_feature_auc.png": "feature_auc.png",
        "29_feature_auc_over_time.png": "feature_auc_time.png", "30_faces_objects.png": "faces_objects.png",
        "31_painting_factors.png": "factors.png", "32_viewing_by_composition_class.png": "comp_class.png",
        }


def copy_figures():
    for src, dst in COPY.items():
        p = os.path.join(AN, "figures", src)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(FIG, dst))


def examples():
    """Teaser: saliency map over the painting (top) and three scanpaths (bottom) for five paintings."""
    picks = ["avatt_0003", "avatt_0019", "avatt_0212", "avatt_0021", "avatt_0216"]  # Scream, Lorrain, Kirchner, crane, La Tour
    picks = [p for p in picks if os.path.exists(os.path.join(DATA, "images", p + ".jpg"))]
    det = json.load(open(os.path.join(AN, "composition", "detections.json")))
    ratios = [det[n]["aspect_ratio"] for n in picks]
    fig, axes = plt.subplots(2, len(picks), figsize=(16, 6.2), gridspec_kw={"width_ratios": ratios, "wspace": 0.03, "hspace": 0.04})
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, name in enumerate(picks):
        x0, y0, x1, y1 = det[name]["crop_box"]
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, "images", name + ".jpg")), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        sm = cv2.imread(os.path.join(DATA, "saliency_maps", name + ".png"), 0)[y0:y1, x0:x1]
        axes[0, k].imshow(img)
        axes[0, k].imshow(sm, cmap="inferno", alpha=0.62)
        axes[1, k].imshow(img)
        with open(os.path.join(DATA, "scanpaths", name + ".jsonl")) as f:
            sps = [json.loads(l) for l in f][:3]
        for sp, c in zip(sps, colors):
            xs = [p["x"] - x0 for p in sp["fixations"] if p["onset_ms"] < 5000]
            ys = [p["y"] - y0 for p in sp["fixations"] if p["onset_ms"] < 5000]
            axes[1, k].plot(xs, ys, "-o", color=c, lw=1.6, ms=4.5, mec="white", mew=0.7, alpha=0.95)
        for ax in axes[:, k]:
            ax.set_axis_off()
            ax.set_xlim(0, x1 - x0)
            ax.set_ylim(y1 - y0, 0)
    save(fig, "examples.png")


def temporal_frames():
    """Frames of the temporal saliency video for one painting."""
    name = "avatt_0003"
    cap = cv2.VideoCapture(os.path.join(DATA, "temporal_saliency_maps", name + ".mp4"))
    frames = {}
    for k in [29, 59, 119, 299]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, fr = cap.read()
        if ok:
            frames[k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    det = json.load(open(os.path.join(AN, "composition", "detections.json")))
    x0, y0, x1, y1 = det[name]["crop_box"]
    fig, axes = plt.subplots(1, len(frames), figsize=(8, 2.9), gridspec_kw={"wspace": 0.04})
    for ax, (k, fr) in zip(axes, frames.items()):
        ax.imshow(fr[y0:y1, x0:x1])
        ax.set_title(f"t = {(k + 1) / 60:.1f} s", fontsize=15, loc="center", fontweight="normal")
        ax.set_axis_off()
    save(fig, "temporal_frames.png")


def movements():
    """Distribution of movements and viewing statistics by movement group."""
    from scipy import stats
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
    meta = meta.rename(columns={"attribution_confidence": "confidence"})
    meta["artist"] = meta.artist.replace("", np.nan)
    fac = pd.read_csv(os.path.join(AN, "per_image_factors.csv"), index_col=0)
    df = fac.join(meta[["movement_group", "confidence", "artist"]])
    counts = meta.movement_group.value_counts()
    M("nMovementGroups", len(counts))
    M("nArtists", meta.artist.dropna().nunique())
    M("nMetaHigh", int((meta.confidence == "high").sum()))
    M("nMetaMedium", int((meta.confidence == "medium").sum()))
    M("nMetaLow", int((meta.confidence == "low").sum()))
    groups = [g for g in counts.index if counts[g] >= 5]
    res = {}
    for o in ["entropy", "human_CC", "fix_dur_ms"]:
        res[o] = stats.kruskal(*[df.loc[df.movement_group == g, o].dropna() for g in groups]).pvalue
    M("kwMovEntropyP", f"{res['entropy']:.1g}")
    M("kwMovCCP", f"{res['human_CC']:.1g}")
    M("kwMovDurP", f"{res['fix_dur_ms']:.1g}")
    order = df.groupby("movement_group").human_CC.median().loc[groups].sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    c = counts.sort_values()
    axes[0].barh(c.index, c.values, color="#2a78d6", edgecolor=SURFACE)
    for i, v in enumerate(c.values):
        axes[0].text(v, i, f" {v}", va="center", fontsize=8, color=INK2)
    axes[0].set_xlabel("Paintings")
    axes[0].set_title("Paintings per movement")
    axes[0].grid(axis="y", visible=False)
    axes[1].boxplot([df.loc[df.movement_group == g, "human_CC"].dropna() for g in order], vert=False,
                    labels=[f"{g} ({counts[g]})" for g in order], showfliers=False, patch_artist=True,
                    medianprops={"color": INK}, boxprops={"facecolor": "#86b6ef", "edgecolor": SURFACE})
    axes[1].set_xlabel("Human split-half CC")
    axes[1].set_title(f"Agreement between observers by movement (Kruskal-Wallis p = {res['human_CC']:.1g})")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "movements.png")


def style():
    r = J("results_style.json")
    vp = r["variance_partitioning"]
    for o, key in [("entropy", "Ent"), ("dispersion_deg", "Disp"), ("agreement_cc", "Agr"), ("fix_dur_ms", "Dur"),
                   ("sacc_amp_deg", "Amp"), ("central_pct", "Cen")]:
        M(f"rFull{key}", f"{vp[o]['R2_full']:.2f}")
        for b, bk in [("Style", "Sty"), ("Composition", "Comp"), ("Content", "Cont"), ("Low-level", "Low"), ("Format", "Fmt")]:
            M(f"u{bk}{key}", f"{vp[o]['blocks'][b]['unique']:.3f}")
            M(f"a{bk}{key}", f"{vp[o]['blocks'][b]['alone_minus_nuisance']:.2f}")
    M("kwStyleAgrP", f"{r['attention_kruskal']['agreement_cc']['p']:.1g}")
    M("kwStyleEntP", f"{r['attention_kruskal']['entropy']['p']:.1g}")
    M("kwStyleCenP", f"{r['attention_kruskal']['central_pct']['p']:.2f}")
    M("cacStyleP", f"{r['cacnet_class_by_style']['p']:.1g}")
    M("faceStyleP", f"{r.get('face_enrichment_kruskal_p', float('nan')):.2f}")
    for src, dst in [("34_attention_by_style.png", "style_attention.png"), ("35_composition_by_style.png", "style_composition.png"),
                     ("36_objects_by_style.png", "style_objects.png"), ("37_features_by_style.png", "style_features.png"),
                     ("38_style_vs_composition_variance.png", "style_variance.png")]:
        shutil.copyfile(os.path.join(AN, "figures", src), os.path.join(FIG, dst))


def main():
    participants()
    movements()
    style()
    dataset()
    temporal()
    for fn in (benchmark, similarity, content):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skipped {fn.__name__}: {e}")
    copy_figures()
    pooled_figures()
    examples()
    temporal_frames()
    with open(os.path.join(GEN, "macros.tex"), "w", encoding="utf-8") as f:
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(macros)} macros, {len(os.listdir(FIG))} figures")


if __name__ == "__main__":
    main()
