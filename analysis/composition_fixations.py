"""Composition vs fixations: spatial and temporal (uses composition_detect + composition_fit outputs).

Usage (from the repository root):  python analysis/composition_fixations.py
Writes figures 24-27 and results_composition.json.

Every comparison is observed vs the other-paintings baseline (see paintfix.py).
  A. Composition elements: share of fixations within 1 deg of the best-fitting template's key points
     (or 0.5 deg of its lines/curves), as log2 enrichment over baseline; paintings that follow the rule
     (top-quartile fit) vs the rest; chosen spiral variant vs the other seven.
  B. Focal subjects and salient objects: enrichment, time course, first-fixation latency, and the
     salient-object map used as a saliency prediction (NSS) over time.
  C. Line following: alignment of saccade direction with the local line orientation of the painting,
     and with the golden-spiral tangent for spiral paintings.
"""
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from analysis import INK, INK2, MUTED, OUT, SERIES, SURFACE, plt, save
from composition_fit import FAM_LABELS, templates
from paintfix import COMP, EPS, baseline, ci95, crop_image, load_all, read_map

WINDOW_S = 1.0
T_MAX = 5.0
FAMILIES = ["thirds", "golden_grid", "centre", "diagonal", "golden_triangles", "dynamic_symmetry",
            "spiral_stretched", "spiral_true_ratio"]


def densify(curve_px, step=2.0):
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(curve_px, axis=0).T))]
    n = max(int(seg[-1] / step), 2)
    t = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(t, seg, curve_px[:, 0]), np.interp(t, seg, curve_px[:, 1])]


def zone_mask(var, w, h, ppd):
    """Boolean crop mask: within 1 deg of key points or 0.5 deg of lines/curves."""
    m = np.zeros((h, w), np.uint8)
    if var["points"] is not None:
        for u, v in var["points"]:
            cv2.circle(m, (int(round(u * w)), int(round(v * h))), int(round(ppd)), 1, -1)
    for c in var["curves"]:
        px = np.round(np.c_[c[:, 0] * w, c[:, 1] * h]).astype(np.int32)
        cv2.polylines(m, [px], False, 1, thickness=max(1, int(round(ppd))))
    return m.astype(bool)


def in_mask(mask, u, v):
    h, w = mask.shape
    xi = np.clip((np.asarray(u) * w).astype(int), 0, w - 1)
    yi = np.clip((np.asarray(v) * h).astype(int), 0, h - 1)
    return mask[yi, xi]


def orientation_field(gray, sigma):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    grad_angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)            # dominant gradient direction (image y down)
    line_angle = -(grad_angle + np.pi / 2)                        # along the edge, math convention (y up)
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coherence = lam / (jxx + jyy + EPS)
    return line_angle, coherence


def saccades(df, w, h, min_px):
    """Saccades between consecutive on-painting fixations of the same scanpath, in this crop's pixels."""
    rows = []
    for _, s in df.groupby(["image", "observer"]):
        s = s[s.on_painting]
        if len(s) < 2:
            continue
        u, v, t = s.u.to_numpy(), s.v.to_numpy(), s.t_s.to_numpy()
        dx, dy = np.diff(u) * w, np.diff(v) * h
        amp = np.hypot(dx, dy)
        keep = amp >= min_px
        rows.append(pd.DataFrame({"mu": ((u[1:] + u[:-1]) / 2)[keep], "mv": ((v[1:] + v[:-1]) / 2)[keep],
                                  "theta": np.arctan2(-dy, dx)[keep], "t_s": t[1:][keep]}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["mu", "mv", "theta", "t_s"])


def alignment(sac, angle, coh):
    if not len(sac):
        return np.nan
    h, w = angle.shape
    xi = np.clip((sac.mu.to_numpy() * w).astype(int), 0, w - 1)
    yi = np.clip((sac.mv.to_numpy() * h).astype(int), 0, h - 1)
    return float(np.mean(coh[yi, xi] * np.cos(2 * (sac.theta.to_numpy() - angle[yi, xi]))))


def first_hit_latency(df, mask):
    """Median over scanpaths of the time of the first fixation inside mask (scanpaths that never hit: excluded)."""
    hits = df[df.on_painting & in_mask(mask, df.u, df.v)]
    first = hits.groupby(["image", "observer"]).t_s.min()
    n_paths = df.groupby(["image", "observer"]).ngroups
    return (float(first.median()) if len(first) else np.nan), (len(first) / n_paths if n_paths else np.nan)


def per_painting(job):
    name, d, fits, obs, base, ppd = job
    w, h = d["crop_size"]
    r = d["aspect_ratio"]
    on, bon = obs[obs.on_painting], base[base.on_painting]
    res = {"image": name}
    T = templates(r)

    # A. composition elements
    for fam in FAMILIES:
        f = fits[name].get(fam, {})
        if f.get("best_variant") is None:
            continue
        variants = T[fam]
        enr = {}
        for vname, var in variants.items():
            m = zone_mask(var, w, h, ppd)
            o, b = in_mask(m, on.u, on.v).mean(), in_mask(m, bon.u, bon.v).mean()
            enr[vname] = np.log2((o + 1e-3) / (b + 1e-3))
            if vname == f["best_variant"]:
                res[f"{fam}__obs"], res[f"{fam}__base"], res[f"{fam}__area"] = o, b, m.mean()
                res[f"{fam}__latency_obs"], res[f"{fam}__reach_obs"] = first_hit_latency(obs, m)
                res[f"{fam}__latency_base"], res[f"{fam}__reach_base"] = first_hit_latency(base, m)
                for t0 in np.arange(0, T_MAX, WINDOW_S):
                    ow = on[(on.t_s >= t0) & (on.t_s < t0 + WINDOW_S)]
                    bw = bon[(bon.t_s >= t0) & (bon.t_s < t0 + WINDOW_S)]
                    res[f"{fam}__tc_obs_{t0:.0f}"] = in_mask(m, ow.u, ow.v).mean() if len(ow) else np.nan
                    res[f"{fam}__tc_base_{t0:.0f}"] = in_mask(m, bw.u, bw.v).mean() if len(bw) else np.nan
        res[f"{fam}__enrich_best"] = enr[f["best_variant"]]
        if len(enr) > 1:
            res[f"{fam}__enrich_others"] = float(np.mean([v for k, v in enr.items() if k != f["best_variant"]]))
        res[f"{fam}__score"] = f.get("score", np.nan)

    # B. focal subjects and the salient-object map
    subj = read_map("subjects", name)
    sod = read_map("sod", name)
    if subj is not None and (subj > 0).any():
        m = subj > 0
        res["subject__obs"], res["subject__base"], res["subject__area"] = in_mask(m, on.u, on.v).mean(), in_mask(m, bon.u, bon.v).mean(), m.mean()
        res["subject__latency_obs"], res["subject__reach_obs"] = first_hit_latency(obs, m)
        res["subject__latency_base"], res["subject__reach_base"] = first_hit_latency(base, m)
        for t0 in np.arange(0, T_MAX, WINDOW_S):
            ow = on[(on.t_s >= t0) & (on.t_s < t0 + WINDOW_S)]
            bw = bon[(bon.t_s >= t0) & (bon.t_s < t0 + WINDOW_S)]
            res[f"subject__tc_obs_{t0:.0f}"] = in_mask(m, ow.u, ow.v).mean() if len(ow) else np.nan
            res[f"subject__tc_base_{t0:.0f}"] = in_mask(m, bw.u, bw.v).mean() if len(bw) else np.nan
    if sod is not None:
        z = (sod.astype(np.float32) - sod.mean()) / (sod.std() + EPS)
        hh, ww = z.shape

        def nss(df):
            if not len(df):
                return np.nan
            return float(z[np.clip((df.v * hh).astype(int), 0, hh - 1), np.clip((df.u * ww).astype(int), 0, ww - 1)].mean())
        res["sod_nss_obs"], res["sod_nss_base"] = nss(on), nss(bon)
        for t0 in np.arange(0, T_MAX, WINDOW_S):
            res[f"sod_nss_obs_{t0:.0f}"] = nss(on[(on.t_s >= t0) & (on.t_s < t0 + WINDOW_S)])
            res[f"sod_nss_base_{t0:.0f}"] = nss(bon[(bon.t_s >= t0) & (bon.t_s < t0 + WINDOW_S)])

    # C. line following
    gray = cv2.cvtColor(crop_image({name: d}, name), cv2.COLOR_BGR2GRAY).astype(np.float32)
    angle, coh = orientation_field(gray, ppd / 2)
    so, sb = saccades(obs, w, h, ppd), saccades(base, w, h, ppd)
    res["lines_align_obs"], res["lines_align_base"] = alignment(so, angle, coh), alignment(sb, angle, coh)
    f = fits[name].get("spiral_stretched", {})
    if f.get("best_variant"):
        curve = T["spiral_stretched"][f["best_variant"]]["curves"][0]
        px = densify(np.c_[curve[:, 0] * w, curve[:, 1] * h], 1.0)
        tang = np.arctan2(-np.gradient(px[:, 1]), np.gradient(px[:, 0]))
        near = np.zeros((h, w), np.int32) - 1
        dist = np.full((h, w), np.inf, np.float32)
        yy, xx = np.mgrid[0:h:4, 0:w:4]
        grid = np.c_[xx.ravel(), yy.ravel()]
        from scipy.spatial import cKDTree
        dd, ii = cKDTree(px).query(grid)
        near_small = ii.reshape(xx.shape)
        dist_small = dd.reshape(xx.shape)
        tang_map = cv2.resize(tang[near_small].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        band = cv2.resize((dist_small <= 2 * ppd).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        res["spiral_align_obs"] = alignment(so, tang_map, band)
        res["spiral_align_base"] = alignment(sb, tang_map, band)
        res["spiral_score"] = f.get("score", np.nan)
    return res


def main():
    det, fix, ppd = load_all()
    with open(os.path.join(COMP, "fits.json")) as f:
        fits = json.load(f)
    names = sorted(det)
    from paintfix import BASE_COLS
    jobs = [(n, det[n], fits, fix.loc[fix.image == n, BASE_COLS], baseline(fix, n, seed=i), ppd)
            for i, n in enumerate(names)]
    with Pool(min(6, max(1, os.cpu_count() - 2))) as pool:
        rows = pool.map(per_painting, jobs)
    df = pd.DataFrame(rows).set_index("image")
    # graded fit: top quartile of the image-only fit score within each family vs bottom quartile
    for fam in FAMILIES:
        if f"{fam}__score" in df:
            sc = df[f"{fam}__score"]
            df[f"{fam}__follows"] = np.where(sc >= sc.quantile(0.75), True, np.where(sc <= sc.quantile(0.25), False, np.nan))
    if "spiral_score" in df:
        sc = df.spiral_score
        df["spiral_follows"] = np.where(sc >= sc.quantile(0.75), True, np.where(sc <= sc.quantile(0.25), False, np.nan))
    df.to_csv(os.path.join(OUT, "per_image_composition_attention.csv"))
    out = {"ppd": round(ppd, 2), "n_paintings": len(df)}

    # --- figure 24: enrichment on composition elements, followers vs the rest
    fam_rows = []
    for fam in FAMILIES:
        if f"{fam}__enrich_best" not in df:
            continue
        fol = df[df[f"{fam}__follows"] == True][f"{fam}__enrich_best"].dropna()
        rest = df[df[f"{fam}__follows"] == False][f"{fam}__enrich_best"].dropna()
        both = df[[f"{fam}__score", f"{fam}__enrich_best"]].dropna()
        rho_fit, p_fit = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])
        p = stats.mannwhitneyu(fol, rest).pvalue if len(fol) > 2 and len(rest) > 2 else np.nan
        p_all = stats.wilcoxon(df[f"{fam}__enrich_best"].dropna()).pvalue
        rec = {"family": fam, "n_follow": len(fol), "rho_fit_vs_enrichment": rho_fit, "p_rho": p_fit, "follow_mean": fol.mean(), "follow_ci": ci95(fol),
               "rest_mean": rest.mean(), "rest_ci": ci95(rest), "p_follow_vs_rest": p,
               "all_mean": df[f"{fam}__enrich_best"].mean(), "p_all_vs_0": p_all}
        if f"{fam}__enrich_others" in df:
            pair = df[[f"{fam}__enrich_best", f"{fam}__enrich_others"]].dropna()
            rec["chosen_minus_other_variants"] = float((pair.iloc[:, 0] - pair.iloc[:, 1]).mean())
            rec["p_chosen_vs_others"] = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue
        fam_rows.append(rec)
    fr = pd.DataFrame(fam_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(fr))
    ax.barh(y - 0.2, fr.follow_mean, 0.38, xerr=fr.follow_ci, color="#2a78d6", label="Best-fitting quarter of paintings",
            error_kw={"ecolor": INK2, "elinewidth": 1})
    ax.barh(y + 0.2, fr.rest_mean, 0.38, xerr=fr.rest_ci, color="#eb6834", label="Worst-fitting quarter",
            error_kw={"ecolor": INK2, "elinewidth": 1})
    for i, rrow in fr.iterrows():
        ax.text(max(rrow.follow_mean + (rrow.follow_ci if not np.isnan(rrow.follow_ci) else 0), rrow.rest_mean + rrow.rest_ci) + 0.03, i,
                f"n={rrow.n_follow} vs {len(df) and int((df[f'{rrow.family}__follows'] == False).sum())}, p={rrow.p_follow_vs_rest:.3g}; "
                f"ρ(fit, enrichment)={rrow.rho_fit_vs_enrichment:.2f}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FAM_LABELS[f] for f in fr.family])
    ax.invert_yaxis()
    ax.set_xlabel("log2 enrichment of fixations on the template's points/lines vs other-paintings baseline")
    ax.set_title("Do viewers fixate composition elements more where the rule fits better? (image-only fit)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, "24_composition_element_enrichment.png")
    out["composition_elements"] = json.loads(fr.round(5).to_json(orient="records"))

    # --- figure 25: time course on focal subjects and composition elements (followers)
    ts = np.arange(0, T_MAX, WINDOW_S)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label, color in [("subject", "Focal subjects", "#2a78d6")] + \
            [(f, FAM_LABELS[f], c) for f, c in [("thirds", "#1baf7a"), ("spiral_stretched", "#eb6834"), ("centre", "#e87ba4")]]:
        cols_o, cols_b = [f"{key}__tc_obs_{t:.0f}" for t in ts], [f"{key}__tc_base_{t:.0f}" for t in ts]
        if cols_o[0] not in df:
            continue
        sub = df if key == "subject" else df[df[f"{key}__follows"] == True]
        if len(sub) < 3:
            continue
        ratio = np.log2((sub[cols_o].to_numpy() + 1e-3) / (sub[cols_b].to_numpy() + 1e-3))
        m, e = np.nanmean(ratio, 0), np.array([ci95(ratio[:, i]) for i in range(len(ts))])
        axes[0].fill_between(ts + 0.5, m - e, m + e, color=color, alpha=0.15, lw=0)
        axes[0].plot(ts + 0.5, m, color=color, marker="o", ms=4, label=f"{label} (n={len(sub)})")
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xlabel("Time since first fixation (s)")
    axes[0].set_ylabel("log2 enrichment vs baseline")
    axes[0].set_title("Attention to subjects and composition over time")
    axes[0].legend(fontsize=8)
    cols_o = [f"sod_nss_obs_{t:.0f}" for t in ts]
    cols_b = [f"sod_nss_base_{t:.0f}" for t in ts]
    for cols, label, color in [(cols_o, "Observed fixations", "#2a78d6"), (cols_b, "Other-paintings baseline", "#eb6834")]:
        a = df[cols].to_numpy()
        m, e = np.nanmean(a, 0), np.array([ci95(a[:, i]) for i in range(len(ts))])
        axes[1].fill_between(ts + 0.5, m - e, m + e, color=color, alpha=0.15, lw=0)
        axes[1].plot(ts + 0.5, m, color=color, marker="o", ms=4, label=label)
    axes[1].set_xlabel("Time since first fixation (s)")
    axes[1].set_ylabel("NSS of the salient-object map")
    axes[1].set_title("Image-predicted salience (BiRefNet) over time")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "25_subjects_composition_over_time.png")

    # --- figure 26: first-fixation latency (observed vs baseline)
    lat_keys = [("subject", "Focal subjects")] + [(f, FAM_LABELS[f]) for f in FAMILIES]
    lat = []
    for key, label in lat_keys:
        if f"{key}__latency_obs" not in df:
            continue
        sub = df if key == "subject" else df[df[f"{key}__follows"] == True]
        pair = sub[[f"{key}__latency_obs", f"{key}__latency_base"]].dropna()
        if len(pair) < 3:
            continue
        diff = pair.iloc[:, 0] - pair.iloc[:, 1]
        lat.append({"key": key, "label": label, "n": len(pair), "obs_median_s": float(pair.iloc[:, 0].median()),
                    "base_median_s": float(pair.iloc[:, 1].median()), "diff_mean_s": float(diff.mean()),
                    "diff_ci": ci95(diff), "p": stats.wilcoxon(diff).pvalue})
    lt = pd.DataFrame(lat)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(range(len(lt)), lt.diff_mean_s, xerr=lt.diff_ci, color="#2a78d6", height=0.6, error_kw={"ecolor": INK2})
    x_text = (lt.diff_mean_s + lt.diff_ci).max() + 0.1
    for i, rrow in lt.iterrows():
        ax.text(x_text, i,
                f"n={rrow.n}, obs {rrow.obs_median_s:.2f}s vs base {rrow.base_median_s:.2f}s, p={rrow.p:.3g}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(lt)))
    ax.set_yticklabels(lt.label)
    ax.invert_yaxis()
    ax.set_xlim(right=x_text + 1.6)
    ax.set_xlabel("Time to first fixation: observed minus baseline (s); negative = reached sooner")
    ax.set_title("How soon do viewers reach subjects and composition elements?")
    ax.grid(axis="y", visible=False)
    save(fig, "26_first_fixation_latency.png")
    out["first_fixation_latency"] = json.loads(lt.round(4).to_json(orient="records"))

    # --- figure 27: line following
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pair = df[["lines_align_obs", "lines_align_base"]].dropna()
    d1 = pair.iloc[:, 0] - pair.iloc[:, 1]
    axes[0].hist(d1, bins=30, color="#2a78d6", edgecolor=SURFACE)
    axes[0].axvline(0, color=INK, lw=0.8)
    axes[0].axvline(d1.mean(), color="#eb6834", lw=1.5, ls="--")
    axes[0].set_title(f"Saccades along the painting's lines\nmean diff {d1.mean():.4f}, Wilcoxon p={stats.wilcoxon(d1).pvalue:.3g}")
    axes[0].set_xlabel("Alignment (observed - baseline) per painting")
    axes[0].set_ylabel("Paintings")
    out["line_following"] = {"mean_obs": float(pair.iloc[:, 0].mean()), "mean_base": float(pair.iloc[:, 1].mean()),
                             "mean_diff": float(d1.mean()), "p": float(stats.wilcoxon(d1).pvalue), "n": len(d1)}
    if "spiral_align_obs" in df:
        groups = []
        for flag, label, color in [(True, "Best-fitting quarter", "#eb6834"), (False, "Worst-fitting quarter", MUTED)]:
            sub = df[df.spiral_follows == flag][["spiral_align_obs", "spiral_align_base"]].dropna()
            d2 = sub.iloc[:, 0] - sub.iloc[:, 1]
            groups.append((label, d2, color))
            out[f"spiral_following_{'follow' if flag else 'rest'}"] = {
                "n": len(d2), "mean_diff": float(d2.mean()) if len(d2) else None,
                "p": float(stats.wilcoxon(d2).pvalue) if len(d2) > 5 else None}
        axes[1].boxplot([g[1] for g in groups], labels=[f"{g[0]}\n(n={len(g[1])})" for g in groups], showfliers=False,
                        patch_artist=True, medianprops={"color": INK}, boxprops={"facecolor": "#f4b393", "edgecolor": SURFACE})
        axes[1].axhline(0, color=INK, lw=0.8)
        axes[1].set_title("Saccades along the best-fitting golden spiral")
        axes[1].set_ylabel("Alignment (observed - baseline)")
        axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "27_line_following.png")

    pair = df[["sod_nss_obs", "sod_nss_base"]].dropna()
    dd = pair.iloc[:, 0] - pair.iloc[:, 1]
    out["salient_object_map_nss"] = {"obs": float(pair.iloc[:, 0].mean()), "base": float(pair.iloc[:, 1].mean()),
                                     "diff": float(dd.mean()), "p": float(stats.wilcoxon(dd).pvalue)}
    pair = df[["subject__obs", "subject__base"]].dropna()
    out["focal_subjects"] = {"n": len(pair), "obs_share": float(pair.iloc[:, 0].mean()), "base_share": float(pair.iloc[:, 1].mean()),
                             "area_share": float(df.subject__area.mean()),
                             "log2_enrichment": float(np.log2((pair.iloc[:, 0] + 1e-3) / (pair.iloc[:, 1] + 1e-3)).mean()),
                             "p": float(stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1]).pvalue)}
    with open(os.path.join(OUT, "results_composition.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
