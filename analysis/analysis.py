"""Descriptive analysis of the AVAtt dataset.

Usage (from the repository root):  python analysis/analysis.py
Reads the dataset (AVATT_DATA, default data/AVAtt) and writes results/figures, results/results.json and
results/per_image.csv (AVATT_RESULTS, default results/).
Paintings are grouped by presentation time ("collection" = 5.5 s or 15 s exposure, from metadata.csv).
"""
import json
import os
import re
import unicodedata

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("AVATT_DATA", os.path.join(ROOT, "data", "AVAtt"))
OUT = os.environ.get("AVATT_RESULTS", os.path.join(ROOT, "results"))
FIG = os.path.join(OUT, "figures")
TP = os.environ.get("AVATT_THIRD_PARTY", os.path.join(ROOT, "third_party"))
os.makedirs(FIG, exist_ok=True)

W, H = 1600, 900
PX_PER_MM = 3.6161          # from Tobii ADCS px vs mm gaze coordinates
SCREEN_W_MM = 1920 / PX_PER_MM
# Reference palette: categorical slots 1-2 (exposure groups), sequential blue ramp, light surface.
SERIES = {"15 s": "#2a78d6", "5.5 s": "#eb6834"}
BLUE = "#2a78d6"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 10,
})


def clean_name(filename):
    stem = os.path.splitext(filename)[0]
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"\s+", "_", stem)
    return re.sub(r"[^A-Za-z0-9_.,()\-]", "", stem)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- loading
def load():
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
    collection = {i: f"{e:g} s" for i, e in meta.exposure_s.items()}
    fix_rows, sac_rows, sp_rows = [], [], []
    for fname in sorted(os.listdir(os.path.join(DATA, "scanpaths"))):
        image = fname[:-6]
        with open(os.path.join(DATA, "scanpaths", fname)) as f:
            for line in f:
                sp = json.loads(line)
                fx = pd.DataFrame(sp["fixations"])
                fx["image"], fx["observer"], fx["collection"] = image, sp["observer"], collection[image]
                fx["distance_mm"] = fx["distance_mm"].astype(float)
                fix_rows.append(fx)
                sp_rows.append({"image": image, "collection": collection[image], "n_fix": len(fx),
                                "viewing_s": (fx.onset_ms.iloc[-1] + fx.duration_ms.iloc[-1]) / 1000})
                if len(fx) < 2:
                    continue
                dist = fx["distance_mm"].ffill().bfill().fillna(fx["distance_mm"].mean()).to_numpy()[1:]
                dx_mm = np.diff(fx.x.to_numpy()) / PX_PER_MM
                dy_mm = np.diff(fx.y.to_numpy()) / PX_PER_MM
                amp = np.degrees(np.arctan(np.hypot(dx_mm, dy_mm) / dist))
                absdir = np.degrees(np.arctan2(-dy_mm, dx_mm)) % 360   # 0 = right, 90 = up
                reldir = np.full_like(absdir, np.nan)
                reldir[1:] = (absdir[1:] - absdir[:-1]) % 360
                sac_rows.append(pd.DataFrame({"collection": collection[image], "amplitude_deg": amp,
                                              "abs_dir_deg": absdir, "rel_dir_deg": reldir}))
    return (pd.concat(fix_rows, ignore_index=True), pd.concat(sac_rows, ignore_index=True),
            pd.DataFrame(sp_rows))


# ---------------------------------------------------------------- image measures
def colorfulness(bgr):
    """Hasler & Suesstrunk (2003) colourfulness on a float BGR array."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def spatial_information(sobel_mag):
    """ITU-T P.910 SI: standard deviation of the Sobel gradient magnitude of luminance."""
    return float(sobel_mag.std())


def sobel_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)[1:-1, 1:-1]


def painting_mask(bgr8, bg=(170, 170, 170), tol=3):
    """True on the painting; False on the uniform grey frame around it (flood-filled from the corners)."""
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for x, y in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]:
        if np.all(np.abs(bgr8[y, x].astype(int) - bg) <= tol) and not mask[y + 1, x + 1]:
            cv2.floodFill(bgr8.copy(), mask, (x, y), 0, (tol,) * 3, (tol,) * 3,
                          cv2.FLOODFILL_MASK_ONLY | 8 | (255 << 8))
    return mask[1:-1, 1:-1] == 0


def image_and_fixation_measures(fix, half):
    """Image-level SI/colourfulness, painting HSV histograms, and per-fixation patch measures.

    All measures are taken on the painting itself; the grey frame around it is excluded.
    """
    per_image, hist_img = [], {"H": np.zeros(180), "S": np.zeros(256), "V": np.zeros(256)}
    fix = fix.copy()
    fix["on_painting"] = False
    for c in ["H", "S", "V", "SI", "colorfulness"]:
        fix[c] = np.nan
    for image, idx in fix.groupby("image").groups.items():
        bgr8 = cv2.imread(os.path.join(DATA, "images", image + ".jpg"))
        paint = painting_mask(bgr8)
        rows, cols = np.where(paint.any(1))[0], np.where(paint.any(0))[0]
        crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        bgr = bgr8.astype(np.float32)
        gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
        sob = np.pad(sobel_magnitude(gray), 1, mode="edge")
        per_image.append({"image": image, "painting_area_pct": round(100 * paint.mean(), 1),
                          "SI": spatial_information(sobel_magnitude(gray[crop])),
                          "colorfulness": colorfulness(bgr[paint])})
        for k, ch, n in [("H", 0, 180), ("S", 1, 256), ("V", 2, 256)]:
            hist_img[k] += np.bincount(hsv[..., ch][paint], minlength=n)[:n]
        xs, ys = fix.loc[idx, "x"].to_numpy(), fix.loc[idx, "y"].to_numpy()
        on = paint[ys, xs]
        fix.loc[idx, "on_painting"] = on
        idx, xs, ys = idx[on], xs[on], ys[on]
        fix.loc[idx, "H"], fix.loc[idx, "S"], fix.loc[idx, "V"] = hsv[ys, xs].T
        si, cf = [], []
        for x, y in zip(xs, ys):
            y0, y1, x0, x1 = max(0, y - half), min(H, y + half), max(0, x - half), min(W, x + half)
            m = paint[y0:y1, x0:x1]
            si.append(sob[y0:y1, x0:x1][m].std())
            cf.append(colorfulness(bgr[y0:y1, x0:x1][m]))
        fix.loc[idx, "SI"], fix.loc[idx, "colorfulness"] = si, cf
    return fix, pd.DataFrame(per_image), hist_img


# ---------------------------------------------------------------- inter-observer congruency
def ioc(fix, sigma):
    """Leave-one-out: share of an observer's fixations inside the Otsu-thresholded map of the others."""
    rows = []
    for image, g in fix.groupby("image"):
        counts = {o: np.zeros((H, W), np.float32) for o in g.observer.unique()}
        for o, s in g.groupby("observer"):
            np.add.at(counts[o], (s.y.to_numpy(), s.x.to_numpy()), 1)
        total = sum(counts.values())
        scores = []
        for o, s in g.groupby("observer"):
            sal = cv2.GaussianBlur(total - counts[o], (0, 0), sigma)
            sal8 = np.uint8(255 * sal / sal.max())
            _, mask = cv2.threshold(sal8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            scores.append(mask[s.y.to_numpy(), s.x.to_numpy()].mean())
        rows.append({"image": image, "IOC": float(np.mean(scores))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plotting helpers
def density_by_collection(ax, df, col, bins, xlabel):
    for label, color in SERIES.items():
        v = df.loc[df.collection == label, col].dropna()
        h, e = np.histogram(v, bins=bins, density=True)
        ax.plot((e[:-1] + e[1:]) / 2, h, color=color, label=f"{label} (median {np.median(v):.0f})"
                if v.median() >= 10 else f"{label} (median {np.median(v):.2f})")
    v = df[col].dropna()
    ax.axvline(v.median(), color=MUTED, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()


def polar_hist(dirs_deg, amps, title, name, rmax=20):
    ok = ~np.isnan(dirs_deg) & ~np.isnan(amps)
    a_bins = np.radians(np.arange(0, 361, 5))
    r_bins = np.arange(0, rmax + 0.5, 0.5)
    hist, _, _ = np.histogram2d(np.radians(dirs_deg[ok]), amps[ok], bins=(a_bins, r_bins))
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 5.4))
    A, R = np.meshgrid(a_bins, r_bins)
    pc = ax.pcolormesh(A, R, hist.T, cmap=SEQ, norm=PowerNorm(0.5))
    ax.set_ylim(0, rmax)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2)
    ax.set_rlabel_position(22.5)
    ax.set_rticks([5, 10, 15, 20])
    ax.set_yticklabels(["5°", "10°", "15°", "20°"])
    cb = fig.colorbar(pc, ax=ax, pad=0.1, shrink=0.8)
    cb.set_label("Saccades per bin (5° × 0.5°)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(title + "\nradius = amplitude (°)", pad=18)
    save(fig, name)


def hue_bar_colors(n):
    return [cv2.cvtColor(np.uint8([[[h, 255, 230]]]), cv2.COLOR_HSV2RGB)[0, 0] / 255 for h in range(n)]


def channel_hist(ax, fix_vals, img_hist, n, title, colors, legend_loc="upper right"):
    fx = np.bincount(fix_vals.astype(int), minlength=n)[:n].astype(float)
    fx /= fx.sum()
    im = img_hist / img_hist.sum()
    ax.bar(np.arange(n), fx, width=1.0, color=colors, linewidth=0, label="Fixated pixels (on the painting)")
    ax.plot(np.arange(n), im, color=INK, lw=1.5, label="All painting pixels")
    ax.set_xlim(0, n)
    ax.set_title(title)
    ax.set_ylabel("Share")
    ax.legend(loc=legend_loc, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.95)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(FIG, exist_ok=True)
    fix, sac, sp = load()
    mean_dist = fix.distance_mm.mean()
    ppd = 2 * mean_dist * np.tan(np.radians(0.5)) * PX_PER_MM
    half = int(round(ppd / 2))  # 1° x 1° patch around each fixation
    print(f"{fix.image.nunique()} images, {len(sp)} scanpaths, {len(fix)} fixations, 1° = {ppd:.2f} px")

    fix, img, hist_img = image_and_fixation_measures(fix, half)
    io = ioc(fix, ppd)
    obs = sp.groupby("image").agg(collection=("collection", "first"), observers=("image", "size"))
    per_image = obs.join(img.set_index("image")).join(io.set_index("image"))
    per_image.round(4).to_csv(os.path.join(OUT, "per_image.csv"))

    # --- 1. observers per image
    fig, ax = plt.subplots(figsize=(7, 3.6))
    counts = per_image.groupby(["observers", "collection"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(counts))
    for label, color in SERIES.items():
        if label in counts:
            ax.bar(counts.index.astype(str), counts[label], bottom=bottom, color=color, label=label,
                   edgecolor=SURFACE, linewidth=2)
            bottom += counts[label].to_numpy()
    ax.set_xlabel("Scanpaths per image")
    ax.set_ylabel("Images")
    ax.set_title(f"Scanpaths per image (n = {len(per_image)} images)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, "01_scanpaths_per_image.png")

    # --- 2. fixation duration, fixations per scanpath, saccade amplitude, screen distance
    fig, ax = plt.subplots(figsize=(7, 3.6))
    density_by_collection(ax, fix, "duration_ms", np.arange(0, 2001, 25), "Fixation duration (ms)")
    ax.set_title("Fixation duration")
    save(fig, "02_fixation_duration.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    density_by_collection(ax, sp, "n_fix", np.arange(0, 81, 2), "Fixations per scanpath")
    ax.set_title("Fixations per scanpath")
    save(fig, "03_fixations_per_scanpath.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    density_by_collection(ax, sac, "amplitude_deg", np.arange(0, 30.5, 0.5), "Saccade amplitude (°)")
    ax.set_title("Saccade amplitude")
    save(fig, "04_saccade_amplitude.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    density_by_collection(ax, fix, "distance_mm", np.arange(400, 801, 5), "Eye–screen distance (mm)")
    ax.axvspan(500, 650, color=GRID, alpha=0.5, lw=0)
    ax.set_title("Viewing distance per fixation")
    save(fig, "05_viewing_distance.png")

    # --- 3. saccade direction x amplitude
    polar_hist(sac.abs_dir_deg.to_numpy(), sac.amplitude_deg.to_numpy(),
               "Absolute saccade direction × amplitude", "06_polar_absolute_direction.png")
    polar_hist(sac.rel_dir_deg.to_numpy(), sac.amplitude_deg.to_numpy(),
               "Relative saccade direction × amplitude", "07_polar_relative_direction.png")

    # --- 4. central bias
    counts_map = np.zeros((H, W), np.float32)
    np.add.at(counts_map, (fix.y.to_numpy(), fix.x.to_numpy()), 1)
    density = cv2.GaussianBlur(counts_map, (0, 0), ppd)
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 2.25), wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(density, cmap=SEQ, extent=(0, W, H, 0), aspect="auto")
    ax.plot(W / 2, H / 2, "+", color=INK, ms=10, mew=1.2)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.grid(False)
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axx.hist(fix.x, bins=np.arange(0, W + 1, 20), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    axx.tick_params(labelbottom=False)
    axx.set_ylabel("Fixations")
    axy = fig.add_subplot(gs[1, 1], sharey=ax)
    axy.hist(fix.y, bins=np.arange(0, H + 1, 20), orientation="horizontal", color=BLUE,
             edgecolor=SURFACE, linewidth=0.5)
    axy.tick_params(labelleft=False)
    axy.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, prune="lower"))
    axy.set_xlabel("Fixations")
    axx.set_title(f"Fixation density over all images (σ = 1° = {ppd:.1f} px)")
    save(fig, "08_central_bias.png")

    # --- 5. image-level colourfulness and SI
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, label in [(axes[0], "colorfulness", "Colourfulness (Hasler & Süsstrunk)"),
                           (axes[1], "SI", "Spatial information (ITU-T P.910)")]:
        ax.hist([per_image.loc[per_image.collection == c, col] for c in SERIES], bins=25, stacked=True,
                color=list(SERIES.values()), label=list(SERIES), edgecolor=SURFACE, linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("Images")
        ax.legend()
    axes[0].set_title("Image colourfulness")
    axes[1].set_title("Image spatial information")
    save(fig, "09_image_colorfulness_SI.png")

    r_p, p_p = stats.pearsonr(per_image.SI, per_image.colorfulness)
    r_s, p_s = stats.spearmanr(per_image.SI, per_image.colorfulness)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color in SERIES.items():
        d = per_image[per_image.collection == label]
        ax.scatter(d.SI, d.colorfulness, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
    ax.set_xlabel("Spatial information (SI)")
    ax.set_ylabel("Colourfulness")
    ax.set_title(f"Image colourfulness vs SI (Pearson r = {r_p:.2f}, Spearman ρ = {r_s:.2f})")
    ax.legend()
    save(fig, "10_image_colorfulness_vs_SI.png")

    # --- 6. HSV at fixations vs whole images
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    channel_hist(axes[0], fix.H.dropna().to_numpy(), hist_img["H"], 180, "Hue (OpenCV 0–179)", hue_bar_colors(180))
    channel_hist(axes[1], fix.S.dropna().to_numpy(), hist_img["S"], 256, "Saturation",
                 [plt.cm.Reds(0.15 + 0.8 * i / 255) for i in range(256)])
    channel_hist(axes[2], fix.V.dropna().to_numpy(), hist_img["V"], 256, "Value",
                 [plt.cm.gray(0.08 + 0.84 * i / 255) for i in range(256)], legend_loc="lower center")
    fig.tight_layout()
    save(fig, "11_hsv_fixations_vs_images.png")

    # --- 7. patch measures at fixations
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    density_by_collection(axes[0], fix, "SI", np.arange(0, 201, 4), "Spatial information of 1° patch")
    axes[0].set_title("SI at fixations")
    density_by_collection(axes[1], fix, "colorfulness", np.arange(0, 151, 3), "Colourfulness of 1° patch")
    axes[1].set_title("Colourfulness at fixations")
    save(fig, "12_fixation_patch_SI_colorfulness.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    hb = ax.hexbin(fix.SI, fix.colorfulness, gridsize=60, cmap=SEQ, mincnt=1, norm=PowerNorm(0.5))
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Fixations", color=INK2)
    cb.outline.set_visible(False)
    rf, _ = stats.spearmanr(fix.SI, fix.colorfulness, nan_policy="omit")
    ax.set_xlabel("Spatial information of 1° patch")
    ax.set_ylabel("Colourfulness of 1° patch")
    ax.set_title(f"Patch colourfulness vs SI at fixations (Spearman ρ = {rf:.2f})")
    ax.grid(False)
    save(fig, "13_fixation_patch_SI_vs_colorfulness.png")

    # --- 8. inter-observer congruency
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist([per_image.loc[per_image.collection == c, "IOC"] for c in SERIES], bins=np.arange(0.15, 1.001, 0.025),
            stacked=True, color=list(SERIES.values()), label=list(SERIES), edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(per_image.IOC.mean(), color=INK, lw=1, ls="--")
    ax.text(per_image.IOC.mean() - 0.01, ax.get_ylim()[1] * 0.97, f"mean {per_image.IOC.mean():.2f}", color=INK2, va="top", ha="right")
    ax.set_xlabel("Inter-observer congruency (leave-one-out, σ = 1°)")
    ax.set_ylabel("Images")
    ax.set_title("Inter-observer congruency per image")
    ax.legend(loc="upper left")
    save(fig, "14_inter_observer_congruency.png")

    # --- results.json
    def summary(f, s, sc, pi):
        return {
            "images": int(pi.shape[0]), "scanpaths": int(len(s)), "fixations": int(len(f)),
            "scanpaths_per_image": {"min": int(pi.observers.min()), "median": float(pi.observers.median()),
                                    "max": int(pi.observers.max())},
            "viewing_time_s_mean": round(s.viewing_s.mean(), 2),
            "fixations_per_scanpath_mean": round(s.n_fix.mean(), 1),
            "fixation_duration_ms": {"mean": round(f.duration_ms.mean(), 1), "median": float(f.duration_ms.median()),
                                     "sd": round(f.duration_ms.std(), 1)},
            "saccade_amplitude_deg": {"mean": round(sc.amplitude_deg.mean(), 2),
                                      "median": round(sc.amplitude_deg.median(), 2),
                                      "sd": round(sc.amplitude_deg.std(), 2)},
            "horizontal_saccades_pct": round(100 * np.mean(np.minimum(sc.abs_dir_deg % 180, 180 - sc.abs_dir_deg % 180) <= 30), 1),
            "viewing_distance_mm": {"mean": round(f.distance_mm.mean(), 1), "sd": round(f.distance_mm.std(), 1),
                                    "pct_500_650": round(100 * f.distance_mm.between(500, 650).mean(), 1)},
            "fixations_in_central_half_pct": round(100 * (f.x.between(W / 4, 3 * W / 4) & f.y.between(H / 4, 3 * H / 4)).mean(), 1),
            "mean_fixation_position_norm": [round(f.x.mean() / W, 3), round(f.y.mean() / H, 3)],
            "image_colorfulness": {"mean": round(pi.colorfulness.mean(), 2), "sd": round(pi.colorfulness.std(), 2)},
            "image_SI": {"mean": round(pi.SI.mean(), 2), "sd": round(pi.SI.std(), 2)},
            "fixations_on_painting_pct": round(100 * f.on_painting.mean(), 1),
            "painting_area_pct_mean": round(pi.painting_area_pct.mean(), 1),
            "fixation_patch_colorfulness_mean": round(f.colorfulness.mean(), 2),
            "fixation_patch_SI_mean": round(f.SI.mean(), 2),
            "IOC": {"mean": round(pi.IOC.mean(), 3), "sd": round(pi.IOC.std(), 3),
                    "min": round(pi.IOC.min(), 3), "max": round(pi.IOC.max(), 3)},
        }

    results = {
        "setup": {"image_px": [W, H], "screen_width_mm": round(SCREEN_W_MM, 1), "px_per_mm": PX_PER_MM,
                  "mean_viewing_distance_mm": round(mean_dist, 1), "px_per_degree": round(ppd, 2),
                  "patch_size_px": 2 * half},
        "all": summary(fix, sp, sac, per_image),
        "image_colorfulness_vs_SI": {"pearson_r": round(r_p, 3), "pearson_p": float(f"{p_p:.3g}"),
                                     "spearman_rho": round(r_s, 3), "spearman_p": float(f"{p_s:.3g}")},
        "by_collection": {c: summary(fix[fix.collection == c], sp[sp.collection == c],
                                     sac[sac.collection == c], per_image[per_image.collection == c])
                          for c in SERIES},
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["all"], indent=2))


if __name__ == "__main__":
    main()
