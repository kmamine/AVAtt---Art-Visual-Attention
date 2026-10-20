"""Fit composition templates to each painting from image content only, with a selection-corrected test.

Usage (from the repository root):  python analysis/composition_fit.py      (after composition_detect.py)
Writes analysis/composition/fits.json, figures 22-23 (template catalogue, contact sheet).

Coordinates: (u, v) in [0, 1] over the painting (u right, v down). Distances use the painting's real
proportions (u scaled by the aspect ratio), expressed as a fraction of the painting diagonal.

Families and variants
  thirds (1), golden grid (1), centre (1), diagonal (2), golden triangles (2),
  dynamic symmetry (1, depends on aspect ratio),
  golden spiral stretched to the canvas (8), golden spiral true ratio, centred golden rectangle (8)

Fit of one variant to one painting's content
  point fit  = salience-weighted proximity of focal subjects to the variant's key points
               (Gaussian, sigma = 5% of the diagonal)
  curve fit  = log2 of the mean "structure" (salient-object probability x edge strength) along the
               variant's lines/curves, relative to the painting average
Selection-corrected test (no fixation data)
  For painting A, every other painting's content is scored on A's templates (null). Each component is
  turned into a percentile against that null, components are averaged, and the best variant is kept.
  The same best-of-variants statistic is computed for every null painting; p = share of null paintings
  doing at least as well. Benjamini-Hochberg q-values across paintings, per family.
"""
import json
import os

import cv2
import numpy as np

from analysis import DATA, FIG, INK, INK2, MUTED, OUT, SURFACE, plt, save

COMP = os.path.join(OUT, "composition")
PHI = (1 + 5 ** 0.5) / 2
GRID_N = 64
SIGMA = 0.05
FDR_Q = 0.10


# ---------------------------------------------------------------- template geometry
def line(p, q, n=60):
    return np.linspace(p, q, n)


def golden_spiral_landscape(turns=14, n_arc=40):
    """Quarter-arc golden spiral inside a phi x 1 rectangle (y down). Returns polyline and eye."""
    x, y, w, h = 0.0, 0.0, PHI, 1.0
    pts = []
    for k in range(turns):
        d = k % 4
        if d == 0:
            s = h; c = (x + s, y + s); a, b = (x, y + s), (x + s, y); x += s; w -= s
        elif d == 1:
            s = w; c = (x, y + s); a, b = (x, y), (x + s, y + s); y += s; h -= s
        elif d == 2:
            s = h; c = (x + w - s, y); a, b = (x + w, y), (x + w - s, y + s); w -= s
        else:
            s = w; c = (x + w, y + h - s); a, b = (x + w, y + h), (x, y + h - s); h -= s
        t0 = np.arctan2(a[1] - c[1], a[0] - c[0])
        t1 = np.arctan2(b[1] - c[1], b[0] - c[0])
        dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
        t = t0 + np.linspace(0, dt, n_arc)
        pts.append(np.c_[c[0] + s * np.cos(t), c[1] + s * np.sin(t)])
    return np.vstack(pts), np.array([x + w / 2, y + h / 2])


def to_norm_rect(pts, rect, r):
    """Map points from a unit golden-rectangle frame (rw x rh) placed at rect=(X, Y, W, H) in real
    painting coordinates (width r, height 1) to normalised (u, v)."""
    X, Y, W, H, rw, rh = rect
    real = np.c_[X + pts[:, 0] / rw * W, Y + pts[:, 1] / rh * H]
    return np.c_[real[:, 0] / r, real[:, 1]]


def spiral_variants(r, mode):
    base, eye = golden_spiral_landscape()
    out = {}
    for orient in ("landscape", "portrait"):
        P, E = (base, eye) if orient == "landscape" else (base[:, ::-1], eye[::-1])
        rw, rh = (PHI, 1.0) if orient == "landscape" else (1.0, PHI)
        if mode == "stretched":
            rect = (0.0, 0.0, r, 1.0, rw, rh)
        else:                                   # largest centred golden rectangle of this orientation
            a = rw / rh
            W, H = (a, 1.0) if r >= a else (r, r / a)
            rect = ((r - W) / 2, (1 - H) / 2, W, H, rw, rh)
        curve, eye_n = to_norm_rect(P, rect, r), to_norm_rect(E[None], rect, r)[0]
        for fx in (False, True):
            for fy in (False, True):
                c, e = curve.copy(), eye_n.copy()
                if fx:
                    c[:, 0], e[0] = 1 - c[:, 0], 1 - e[0]
                if fy:
                    c[:, 1], e[1] = 1 - c[:, 1], 1 - e[1]
                name = f"{orient}{'-flipH' if fx else ''}{'-flipV' if fy else ''}"
                out[name] = {"points": e[None], "curves": [c]}
    return out


def foot(p, a, b):
    """Foot of the perpendicular from p onto line ab (real coordinates)."""
    d = b - a
    return a + d * np.dot(p - a, d) / np.dot(d, d)


def templates(r):
    """All families -> variants -> {'points': (k, 2) or None, 'curves': [polylines]} in (u, v)."""
    def norm(P):
        return np.c_[P[:, 0] / r, P[:, 1]]

    T = {}
    g = 1 / PHI ** 2
    for fam, (lo, hi) in {"thirds": (1 / 3, 2 / 3), "golden_grid": (g, 1 - g)}.items():
        pts = np.array([[lo, lo], [hi, lo], [lo, hi], [hi, hi]])
        curves = [line([lo, 0], [lo, 1]), line([hi, 0], [hi, 1]), line([0, lo], [1, lo]), line([0, hi], [1, hi])]
        T[fam] = {"grid": {"points": pts, "curves": [np.array(c) for c in curves]}}
    T["centre"] = {"centre": {"points": np.array([[0.5, 0.5]]), "curves": []}}
    T["diagonal"] = {"rising": {"points": None, "curves": [line([0, 1], [1, 0])]},
                     "falling": {"points": None, "curves": [line([0, 0], [1, 1])]}}

    C = {"tl": np.array([0.0, 0.0]), "tr": np.array([r, 0.0]), "bl": np.array([0.0, 1.0]), "br": np.array([r, 1.0])}
    T["golden_triangles"] = {}
    for name, (a, b, o1, o2) in {"rising": ("bl", "tr", "tl", "br"), "falling": ("tl", "br", "tr", "bl")}.items():
        f1, f2 = foot(C[o1], C[a], C[b]), foot(C[o2], C[a], C[b])
        curves = [norm(line(C[a], C[b])), norm(line(C[o1], f1)), norm(line(C[o2], f2))]
        T["golden_triangles"][name] = {"points": norm(np.array([f1, f2])), "curves": curves}

    # dynamic symmetry: both diagonals + the four reciprocals (perpendicular to a diagonal through a corner)
    curves, eyes = [], []
    for a, b in (("bl", "tr"), ("tl", "br")):
        curves.append(norm(line(C[a], C[b])))
        for o in [k for k in C if k not in (a, b)]:
            f = foot(C[o], C[a], C[b])
            d = f - C[o]
            # extend the reciprocal from the corner through the foot until it meets the frame
            ts = [t for t in [(-C[o][0]) / d[0] if d[0] else np.inf, (r - C[o][0]) / d[0] if d[0] else np.inf,
                              (-C[o][1]) / d[1] if d[1] else np.inf, (1 - C[o][1]) / d[1] if d[1] else np.inf] if t > 1e-9]
            end = C[o] + d * min(ts)
            curves.append(norm(line(C[o], end)))
            eyes.append(f)
    T["dynamic_symmetry"] = {"grid": {"points": norm(np.unique(np.round(np.array(eyes), 6), axis=0)), "curves": curves}}
    T["spiral_stretched"] = spiral_variants(r, "stretched")
    T["spiral_true_ratio"] = spiral_variants(r, "true")
    return T


# ---------------------------------------------------------------- content per painting
def resample(curve, n=120):
    """Evenly spaced points along a polyline (in normalised coordinates)."""
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(curve, axis=0).T))]
    if seg[-1] == 0:
        return curve[:1]
    t = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(t, seg, curve[:, 0]), np.interp(t, seg, curve[:, 1])]


def content(name, det):
    d = det[name]
    w, h = d["crop_size"]
    crop = cv2.imread(os.path.join(DATA, "images", name + ".jpg"))
    x0, y0, x1, y1 = d["crop_box"]
    crop = crop[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    sod = cv2.imread(os.path.join(COMP, "sod", name + ".png"), 0).astype(np.float32) / 255
    s = cv2.resize(edges / (edges.mean() + 1e-9) * (0.25 + sod), (GRID_N, GRID_N), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), GRID_N * 0.02)
    s = s / (s.mean() + 1e-9)
    subj = d["subjects"]
    pts = np.array([[t["cx"] / w, t["cy"] / h] for t in subj]) if subj else np.zeros((0, 2))
    wts = np.array([t["salience_share"] for t in subj]) if subj else np.zeros(0)
    if len(wts):
        wts = wts / wts.sum()
    return {"S": s, "pts": pts, "wts": wts}


def bilinear(S, uv):
    x = np.clip(uv[:, 0] * (GRID_N - 1), 0, GRID_N - 1)
    y = np.clip(uv[:, 1] * (GRID_N - 1), 0, GRID_N - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, GRID_N - 1), np.minimum(y0 + 1, GRID_N - 1)
    fx, fy = x - x0, y - y0
    return (S[y0, x0] * (1 - fx) * (1 - fy) + S[y0, x1] * fx * (1 - fy) + S[y1, x0] * (1 - fx) * fy + S[y1, x1] * fx * fy)


def scores(cont, var, r):
    """(point fit or nan, curve fit or nan) of one content on one template variant."""
    diag = np.hypot(r, 1)
    pf = np.nan
    if var["points"] is not None and len(cont["pts"]):
        d = np.hypot((cont["pts"][:, None, 0] - var["points"][None, :, 0]) * r,
                     cont["pts"][:, None, 1] - var["points"][None, :, 1]) / diag
        pf = float((cont["wts"] * np.exp(-d.min(1) ** 2 / (2 * SIGMA ** 2))).sum())
    cf = np.nan
    if var["curves"]:
        if "samples" not in var:
            smp = np.vstack([resample(c) for c in var["curves"]])
            var["samples"] = smp[(smp[:, 0] >= 0) & (smp[:, 0] <= 1) & (smp[:, 1] >= 0) & (smp[:, 1] <= 1)]
        cf = float(np.log2(bilinear(cont["S"], var["samples"]).mean() + 1e-9))
    return pf, cf


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def pct(x, null):
    null = null[~np.isnan(null)]
    if np.isnan(x) or not len(null):
        return np.nan
    return float((np.sum(null < x) + 0.5 * np.sum(null == x)) / len(null))


def main():
    with open(os.path.join(COMP, "detections.json")) as f:
        det = json.load(f)
    names = sorted(det)
    conts = {n: content(n, det) for n in names}
    fits = {n: {} for n in names}
    families = list(templates(1.5))

    for A in names:
        r = det[A]["aspect_ratio"]
        T = templates(r)
        for fam in families:
            V = T[fam]
            vnames = list(V)
            # raw scores: rows = paintings (content), cols = variants, 2 components
            raw = np.array([[scores(conts[B], V[v], r) for v in vnames] for B in names])   # (N, V, 2)
            ia = names.index(A)
            others = np.array([i for i in range(len(names)) if i != ia])
            comb = np.full((len(names), len(vnames)), np.nan)
            for j in range(len(vnames)):
                for c in range(2):
                    null = raw[others, j, c]
                    col = np.array([pct(raw[i, j, c], null if i == ia else np.delete(null, np.where(others == i)[0]))
                                    for i in range(len(names))])
                    comb[:, j] = np.where(np.isnan(comb[:, j]), col, np.where(np.isnan(col), comb[:, j],
                                                                             (comb[:, j] + col) / 2))
            best = np.nanargmax(comb[ia]) if not np.all(np.isnan(comb[ia])) else None
            if best is None:
                fits[A][fam] = {"best_variant": None, "p": None}
                continue
            stat_a = comb[ia, best]
            null_max = np.nanmax(comb[others], axis=1)
            p = (1 + np.sum(null_max >= stat_a)) / (1 + len(others))
            fits[A][fam] = {"best_variant": vnames[best], "score": round(float(stat_a), 4), "p": round(float(p), 4),
                            "point_fit": None if np.isnan(raw[ia, best, 0]) else round(float(raw[ia, best, 0]), 4),
                            "curve_fit": None if np.isnan(raw[ia, best, 1]) else round(float(raw[ia, best, 1]), 4),
                            "n_variants": len(vnames)}
        print(f"  fitted {A}", flush=True)

    for fam in families:
        ok = [n for n in names if fits[n][fam].get("p") is not None]
        q = bh([fits[n][fam]["p"] for n in ok])
        for n, qq in zip(ok, q):
            fits[n][fam]["q"] = round(float(qq), 4)
            fits[n][fam]["follows"] = bool(qq < FDR_Q)

    with open(os.path.join(COMP, "fits.json"), "w") as f:
        json.dump(fits, f, indent=1)
    summary = {fam: {"paintings_following (q<0.10)": int(sum(fits[n][fam].get("follows", False) for n in names)),
                     "paintings_p<0.05": int(sum((fits[n][fam].get("p") or 1) < 0.05 for n in names))}
               for fam in families}
    print(json.dumps(summary, indent=1))
    template_figure()
    contact_sheet(det, fits, names)


# ---------------------------------------------------------------- figures
FAM_COLORS = {"thirds": "#2a78d6", "golden_grid": "#1baf7a", "centre": "#e87ba4", "diagonal": "#eda100",
              "golden_triangles": "#6250d6", "dynamic_symmetry": "#008300",
              "spiral_stretched": "#eb6834", "spiral_true_ratio": "#e34948"}
FAM_LABELS = {"thirds": "Rule of thirds", "golden_grid": "Golden grid", "centre": "Centre", "diagonal": "Diagonal",
              "golden_triangles": "Golden triangles", "dynamic_symmetry": "Dynamic symmetry",
              "spiral_stretched": "Golden spiral (stretched)", "spiral_true_ratio": "Golden spiral (true ratio)"}


def draw_variant(ax, var, r, w, h, color, lw=1.6):
    for c in var["curves"]:
        ax.plot(c[:, 0] * w, c[:, 1] * h, color=color, lw=lw, alpha=0.9)
    if var["points"] is not None:
        ax.scatter(var["points"][:, 0] * w, var["points"][:, 1] * h, s=36, color=color, edgecolor="white", lw=1, zorder=5)


def template_figure(r=1.45):
    T = templates(r)
    panels = [(f, v) for f in ["thirds", "golden_grid", "centre", "dynamic_symmetry"] for v in T[f]] + \
             [("diagonal", v) for v in T["diagonal"]] + [("golden_triangles", v) for v in T["golden_triangles"]] + \
             [("spiral_stretched", v) for v in T["spiral_stretched"]] + [("spiral_true_ratio", v) for v in list(T["spiral_true_ratio"])[:2]]
    cols = 6
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), gridspec_kw={"hspace": 0.5})
    for ax, (f, v) in zip(axes.ravel(), panels):
        ax.add_patch(plt.Rectangle((0, 0), r, 1, fill=True, color="#f0efec", lw=0))
        draw_variant(ax, T[f][v], r, r, 1, FAM_COLORS[f], lw=1.4)
        ax.set_xlim(-0.02, r + 0.02)
        ax.set_ylim(1.02, -0.02)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{FAM_LABELS[f]}\n{v}", fontsize=8, loc="center", fontweight="normal")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Composition templates and their variants (canvas aspect ratio {r}:1)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "22_composition_templates.png")


def contact_sheet(det, fits, names, n=20, seed=1):
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(names, n, replace=False))
    fig, axes = plt.subplots(4, 5, figsize=(20, 17))
    for ax, name in zip(axes.ravel(), pick):
        d = det[name]
        x0, y0, x1, y1 = d["crop_box"]
        w, h = d["crop_size"]
        r = d["aspect_ratio"]
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, "images", name + ".jpg"))[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        sod = cv2.imread(os.path.join(COMP, "sod", name + ".png"), 0)
        ax.contour(sod > 127, levels=[0.5], colors=["#00e5ff"], linewidths=1.2)
        for k, s in enumerate(d["subjects"], 1):
            bx = s["box"]
            ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, color="#ffd400", lw=1.5))
            ax.text(bx[0] + 4, bx[1] + 4, f"{k} {s['label']}", color="black", fontsize=7, va="top",
                    bbox={"facecolor": "#ffd400", "edgecolor": "none", "pad": 1})
        for fc in d["faces"]:
            ax.add_patch(plt.Circle((fc["x"], fc["y"]), fc["r"], fill=False, color="#ff3df5", lw=1.3))
        if d.get("horizon_v") is not None:
            ax.axhline(d["horizon_v"] * h, color="white", lw=1, ls="--")
        fam_ok = [(fits[name][f]["p"], f) for f in fits[name] if fits[name][f].get("p") is not None]
        p_best, fam = min(fam_ok)
        T = templates(r)
        draw_variant(ax, T[fam][fits[name][fam]["best_variant"]], r, w, h, FAM_COLORS[fam], lw=2)
        cac = max(d["composition_probs"]["cacnet"], key=d["composition_probs"]["cacnet"].get)
        clip = max(d["composition_probs"]["clip"], key=d["composition_probs"]["clip"].get)
        q = fits[name][fam].get("q")
        ax.set_title(f"{name[:34]}\nbest fit: {FAM_LABELS[fam]} ({fits[name][fam]['best_variant']}), p={p_best:.3f}, q={q:.2f}\n"
                     f"CACNet: {cac}   CLIP: {clip}", fontsize=8, loc="left", fontweight="normal", color=INK)
        ax.axis("off")
    fig.suptitle("Contact sheet: salient objects (cyan), focal subjects (yellow), faces (magenta), horizon (dashed), "
                 "best-fitting composition template (colour)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "23_composition_contact_sheet.png")


if __name__ == "__main__":
    main()
