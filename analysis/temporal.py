"""Temporal dynamics of viewing in AVAtt (first 5 s, the viewing time shared by every painting).

Usage (from the repository root):  python analysis/temporal.py
Writes figures 17-19 and results_temporal.json next to this script.
"""
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from analysis import H, INK, OUT, PX_PER_MM, W, load, plt, save
from paintfix import ci95

SCALE = 2
h, w = H // SCALE, W // SCALE
EPS = np.finfo(float).eps
BIN_S = 0.5          # bins for fixation duration / saccade amplitude
WINDOW_S = 1.0       # windows for inter-observer agreement
STEP_S = 0.25        # steps for map convergence
T_MAX = 5.0          # shortest presentation time: every painting is analysed over the same window
BLUE, ORANGE = "#2a78d6", "#eb6834"


def counts_map(xs, ys):
    m = np.zeros((h, w), np.float32)
    np.add.at(m, (ys // SCALE, xs // SCALE), 1)
    return m


def nss(sal, xs, ys):
    z = (sal - sal.mean()) / (sal.std() + EPS)
    return float(z[ys // SCALE, xs // SCALE].mean())


def cc(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def with_saccades(fix):
    """Add the amplitude (deg) of the saccade that landed on each fixation."""
    fix = fix.sort_values(["image", "observer", "onset_ms"]).copy()
    dist = fix.distance_mm.fillna(fix.distance_mm.mean()).to_numpy()
    dx = fix.groupby(["image", "observer"]).x.diff().to_numpy() / PX_PER_MM
    dy = fix.groupby(["image", "observer"]).y.diff().to_numpy() / PX_PER_MM
    fix["amplitude_deg"] = np.degrees(np.arctan(np.hypot(dx, dy) / dist))
    fix["t_s"] = fix.onset_ms / 1000
    return fix


def image_dynamics(job):
    image, g, sigma = job
    obs = {o: s for o, s in g.groupby("observer")}
    rows_ioc, rows_conv = [], []
    for t0 in np.arange(0, T_MAX, WINDOW_S):
        win = {o: s[(s.t_s >= t0) & (s.t_s < t0 + WINDOW_S)] for o, s in obs.items()}
        win = {o: s for o, s in win.items() if len(s)}
        if len(win) < 3:
            continue
        cmaps = {o: counts_map(s.x.to_numpy(), s.y.to_numpy()) for o, s in win.items()}
        total = sum(cmaps.values())
        scores = [nss(cv2.GaussianBlur(total - cmaps[o], (0, 0), sigma), s.x.to_numpy(), s.y.to_numpy())
                  for o, s in win.items()]
        rows_ioc.append({"image": image, "t0": t0, "NSS_loo": np.mean(scores)})
    final = cv2.GaussianBlur(counts_map(g.x.to_numpy(), g.y.to_numpy()), (0, 0), sigma)
    for t in np.arange(STEP_S, T_MAX + 1e-9, STEP_S):
        part = g[g.t_s < t]
        if part.empty:
            continue
        m = cv2.GaussianBlur(counts_map(part.x.to_numpy(), part.y.to_numpy()), (0, 0), sigma)
        rows_conv.append({"image": image, "t": t, "CC_to_final": cc(m, final)})
    return rows_ioc, rows_conv


def binned(fix, col):
    """Mean of `col` per time bin: averaged within scanpath first, then across scanpaths."""
    d = fix.dropna(subset=[col]).copy()
    d["bin"] = (d.t_s // BIN_S) * BIN_S + BIN_S / 2
    sp = d.groupby(["image", "observer", "bin"])[col].mean().reset_index()
    g = sp.groupby("bin")[col]
    return g.mean(), g.apply(ci95)


def curve(ax, m, e, color, label=None):
    ax.fill_between(m.index, m - e, m + e, color=color, alpha=0.18, lw=0)
    ax.plot(m.index, m, color=color, marker="o", ms=3, label=label)


def main():
    fix, _, _ = load()
    fix = with_saccades(fix)
    fix = fix[fix.t_s < T_MAX]
    ppd = 2 * fix.distance_mm.mean() * np.tan(np.radians(0.5)) * PX_PER_MM
    sigma = ppd / SCALE

    # --- figure 17: fixation duration and saccade amplitude over time
    dur_m, dur_e = binned(fix, "duration_ms")
    amp_m, amp_e = binned(fix, "amplitude_deg")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    curve(axes[0], dur_m, dur_e, BLUE)
    axes[0].set_ylabel("Mean fixation duration (ms)")
    axes[0].set_title("Fixation duration")
    curve(axes[1], amp_m, amp_e, ORANGE)
    axes[1].set_ylabel("Mean saccade amplitude (°)")
    axes[1].set_title("Saccade amplitude")
    for ax in axes:
        ax.set_xlabel("Time since first fixation (s)")
    fig.tight_layout()
    save(fig, "17_duration_amplitude_over_time.png")

    jobs = [(im, g[["observer", "x", "y", "t_s"]], sigma) for im, g in fix.groupby("image")]
    ioc, conv = [], []
    with Pool(min(6, max(1, os.cpu_count() - 2))) as pool:
        for a, b in pool.imap_unordered(image_dynamics, jobs):
            ioc.extend(a)
            conv.extend(b)
    ioc, conv = pd.DataFrame(ioc), pd.DataFrame(conv)

    # --- figure 18: inter-observer agreement per time window
    g = ioc.groupby("t0").NSS_loo
    m, e = g.mean(), g.apply(ci95)
    m.index, e.index = m.index + WINDOW_S / 2, e.index + WINDOW_S / 2
    fig, ax = plt.subplots(figsize=(5, 3.4))
    curve(ax, m, e, BLUE)
    ax.set_xlabel("Time since first fixation (s)")
    ax.set_ylabel("Leave-one-out NSS")
    ax.set_title(f"Inter-observer agreement ({WINDOW_S:.0f}-s windows)")
    save(fig, "18_agreement_over_time.png")

    # --- figure 19: convergence of the cumulative map to the 5-s map
    g = conv.groupby("t").CC_to_final
    m2, e2 = g.mean(), g.apply(ci95)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    curve(ax, m2, e2, BLUE)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time since first fixation (s)")
    ax.set_ylabel(f"CC with the {T_MAX:.0f}-s map")
    ax.set_title("Convergence of the saliency map")
    save(fig, "19_map_convergence.png")

    reach = {f"CC_{q}": float(m2.index[np.argmax(m2.to_numpy() >= q)]) if (m2 >= q).any() else None
             for q in (0.8, 0.9)}
    out = {
        "setup": {"window_s": T_MAX, "bin_s": BIN_S, "agreement_window_s": WINDOW_S, "step_s": STEP_S,
                  "time_zero": "onset of the first fixation on the image", "sigma_px_full_res": round(ppd, 2),
                  "n_images": int(fix.image.nunique())},
        "fixation_duration_ms_by_bin": {f"{k:.2f}": round(v, 2) for k, v in dur_m.items()},
        "saccade_amplitude_deg_by_bin": {f"{k:.2f}": round(v, 3) for k, v in amp_m.items()},
        "NSS_loo_by_window": {f"{k:.2f}": round(v, 3) for k, v in m.items()},
        "CC_to_final_by_time": {f"{k:.2f}": round(v, 4) for k, v in m2.items()},
        "time_to_reach_CC": reach,
        "first_5s": {"fixations_per_scanpath": round(fix.groupby(["image", "observer"]).size().mean(), 2),
                     "fixation_duration_ms_median": float(fix.duration_ms.median()),
                     "saccade_amplitude_deg_median": round(float(fix.amplitude_deg.median()), 2)},
    }
    with open(os.path.join(OUT, "results_temporal.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("NSS_loo_by_window", "time_to_reach_CC", "first_5s")}, indent=1))


if __name__ == "__main__":
    main()
