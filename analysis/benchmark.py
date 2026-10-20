"""Saliency benchmark on the paintings: human agreement, saliency models, centre-bias baselines,
and observer convergence.

Usage (from the repository root):  python analysis/benchmark.py
  (after composition_detect.py for the painting boxes, and saliency_models.py for the model maps)
Writes figures 15-16, results_benchmark.json and per_image_human_agreement.csv next to this script.

Everything is evaluated on the painting only (grey frame removed), at half resolution, with fixations
that landed on the painting and a Gaussian of 1 degree for the human maps.
Predictors, all scored against the same held-out half of the observers (10 random splits per painting):
  Human (other half), Single observer, the saliency models in saliency_models/, the dataset centre
  bias (other paintings' fixations in painting coordinates) and a Gaussian centre bias.
"""
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from analysis import INK, INK2, OUT, PX_PER_MM, SERIES, plt, save
from paintfix import load_all

SCALE = 2
N_SPLITS = 10
N_REPEATS_CONVERGENCE = 10
HELD_OUT = 5
K_MAX = 9            # images with >= 14 observers
EPS = np.finfo(float).eps
RNG_SEED = 0
NORM_GRID = 128      # resolution of the painting-coordinate centre prior
SAL = os.path.join(OUT, "saliency_models")
MODEL_LABELS = {"deepgaze_iie": "DeepGaze IIE", "transalnet_res": "TranSalNet-Res", "unisal": "UNISAL",
                "gbvs": "GBVS", "ittikoch": "Itti-Koch", "spectral_residual": "Spectral residual",
                "salient_objects": "Salient objects (BiRefNet)"}
ORDER = (["Human (other half)", "Single observer"] + list(MODEL_LABELS.values())
         + ["Centre bias (dataset)", "Centre bias (Gaussian)"])
COLORS = {"Human (other half)": "#2a78d6", "Single observer": "#86b6ef",
          "DeepGaze IIE": "#1baf7a", "TranSalNet-Res": "#1baf7a", "UNISAL": "#1baf7a",
          "GBVS": "#6250d6", "Itti-Koch": "#6250d6", "Spectral residual": "#6250d6",
          "Salient objects (BiRefNet)": "#eda100",
          "Centre bias (dataset)": "#eb6834", "Centre bias (Gaussian)": "#f4b393"}


# ---------------------------------------------------------------- maps
def fixation_counts(cx, cy, shape):
    m = np.zeros(shape, np.float32)
    np.add.at(m, (cy, cx), 1)
    return m


def blur(counts, sigma):
    return cv2.GaussianBlur(counts, (0, 0), sigma)


def norm_hist(u, v):
    m = np.zeros((NORM_GRID, NORM_GRID), np.float32)
    np.add.at(m, (np.clip((v * NORM_GRID).astype(int), 0, NORM_GRID - 1),
                  np.clip((u * NORM_GRID).astype(int), 0, NORM_GRID - 1)), 1)
    return m


# ---------------------------------------------------------------- metrics (Bylinskii et al., 2019)
def auc_judd(sal, fix_rc):
    s = sal.ravel().astype(np.float64)
    s = (s - s.min()) / (s.max() - s.min() + EPS)
    s = s + np.random.default_rng(0).random(s.size) * 1e-7      # break ties, as in the MIT benchmark code
    f = s[fix_rc]
    thresholds = np.sort(f)[::-1]
    n_fix, n_pix = len(f), s.size
    sorted_s = np.sort(s)
    above = n_pix - np.searchsorted(sorted_s, thresholds, side="left")
    tp = np.concatenate([[0], np.arange(1, n_fix + 1) / n_fix, [1]])
    fp = np.concatenate([[0], (above - np.arange(1, n_fix + 1)) / (n_pix - n_fix), [1]])
    return float(np.trapz(tp, fp))


def auc_ranks(pos, neg):
    """Probability that a positive scores above a negative (Mann-Whitney AUC)."""
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def shuffled_auc(sal, fix_rc, other_rc):
    s = sal.ravel()
    return auc_ranks(s[fix_rc], s[other_rc])


def nss(sal, fix_rc):
    s = sal.ravel()
    return float(((s - s.mean()) / (s.std() + EPS))[fix_rc].mean())


def cc(sal, gt):
    a, b = sal.ravel() - sal.mean(), gt.ravel() - gt.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def sim(sal, gt):
    a, b = sal / (sal.sum() + EPS), gt / (gt.sum() + EPS)
    return float(np.minimum(a, b).sum())


def kld(sal, gt):
    a, b = sal / (sal.sum() + EPS), gt / (gt.sum() + EPS)
    return float((b * np.log(EPS + b / (a + EPS))).sum())


def all_metrics(sal, gt_map, fix_rc, other_rc):
    return {"AUC-Judd": auc_judd(sal, fix_rc), "sAUC": shuffled_auc(sal, fix_rc, other_rc),
            "NSS": nss(sal, fix_rc), "CC": cc(sal, gt_map), "SIM": sim(sal, gt_map), "KL": kld(sal, gt_map)}


# ---------------------------------------------------------------- per-image work
_SHARED = {}


def init_worker(total_norm, sigma, model_files):
    _SHARED.update(total=total_norm, sigma=sigma,
                   models={k: np.load(p, allow_pickle=False) for k, p in model_files.items()})


def evaluate_image(job):
    image, collection, shape, observers, own_norm, other_rc, seed = job
    sigma = _SHARED["sigma"]
    hc, wc = shape
    rng = np.random.default_rng(seed)
    obs_ids = list(observers)
    n = len(obs_ids)
    counts = {o: fixation_counts(*observers[o], shape) for o in obs_ids}
    centre_prior = blur(cv2.resize(_SHARED["total"] - own_norm, (wc, hc), interpolation=cv2.INTER_LINEAR), sigma)
    yy, xx = np.mgrid[0:hc, 0:wc]
    gauss_prior = np.exp(-(((xx - wc / 2) / (wc / 4)) ** 2 + ((yy - hc / 2) / (hc / 4)) ** 2) / 2).astype(np.float32)
    fixed = {"Centre bias (dataset)": centre_prior, "Centre bias (Gaussian)": gauss_prior}
    for key, z in _SHARED["models"].items():
        m = z[image].astype(np.float32)
        fixed[MODEL_LABELS[key]] = m if m.shape == shape else cv2.resize(m, (wc, hc), interpolation=cv2.INTER_LINEAR)
    rows = []

    # split-half: every predictor is scored against the same held-out half
    for split in range(N_SPLITS):
        perm = rng.permutation(obs_ids)
        a, b = perm[: n // 2], perm[n // 2:]
        gt_counts = sum(counts[o] for o in b)
        gt_map = blur(gt_counts, sigma)
        fix_rc = np.flatnonzero(gt_counts.ravel())
        fix_rc = np.repeat(fix_rc, gt_counts.ravel()[fix_rc].astype(int))
        predictors = {"Human (other half)": blur(sum(counts[o] for o in a), sigma),
                      "Single observer": blur(counts[rng.choice(a)], sigma), **fixed}
        for name, sal in predictors.items():
            rows.append({"image": image, "collection": collection, "split": split, "model": name,
                         **all_metrics(sal, gt_map, fix_rc, other_rc)})

    # convergence: a fixed held-out group of 5 vs maps built from k other observers
    conv = []
    for rep in range(N_REPEATS_CONVERGENCE):
        perm = rng.permutation(obs_ids)
        held, pool = perm[:HELD_OUT], perm[HELD_OUT:]
        gt_counts = sum(counts[o] for o in held)
        gt_map = blur(gt_counts, sigma)
        fix_rc = np.flatnonzero(gt_counts.ravel())
        fix_rc = np.repeat(fix_rc, gt_counts.ravel()[fix_rc].astype(int))
        acc = np.zeros(shape, np.float32)
        for k, o in enumerate(pool, 1):
            acc += counts[o]
            sal = blur(acc, sigma)
            conv.append({"image": image, "collection": collection, "rep": rep, "k": k,
                         "CC": cc(sal, gt_map), "NSS": nss(sal, fix_rc), "AUC-Judd": auc_judd(sal, fix_rc)})
    return rows, conv


def main():
    det, fix, ppd = load_all()
    sigma = ppd / SCALE
    on = fix[fix.on_painting].copy()
    print(f"1 deg = {ppd:.2f} px; painting-only evaluation at half resolution, sigma = {sigma:.2f} px; "
          f"{len(on)} of {len(fix)} fixations on the paintings", flush=True)
    total_norm = norm_hist(on.u.to_numpy(), on.v.to_numpy())
    model_files = {k: os.path.join(SAL, k + ".npz") for k in MODEL_LABELS if os.path.exists(os.path.join(SAL, k + ".npz"))}
    print("models:", ", ".join(model_files) or "none", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    jobs = []
    for i, (image, g) in enumerate(on.groupby("image")):
        x0, y0, x1, y1 = det[image]["crop_box"]
        wc, hc = max((x1 - x0) // SCALE, 1), max((y1 - y0) // SCALE, 1)
        cx = lambda s: np.clip((s.u.to_numpy() * wc).astype(int), 0, wc - 1)
        cy = lambda s: np.clip((s.v.to_numpy() * hc).astype(int), 0, hc - 1)
        observers = {o: (cx(s), cy(s)) for o, s in g.groupby("observer")}
        other = on[on.image != image]
        other = other.iloc[rng.choice(len(other), size=min(len(other), 20000), replace=False)]
        other_rc = cy(other) * wc + cx(other)
        jobs.append((image, g.collection.iloc[0], (hc, wc), observers,
                     norm_hist(g.u.to_numpy(), g.v.to_numpy()), other_rc, i))

    rows, conv = [], []
    with Pool(min(6, max(1, os.cpu_count() - 2)), initializer=init_worker,
              initargs=(total_norm, sigma, model_files)) as pool:
        for k, (r, c) in enumerate(pool.imap_unordered(evaluate_image, jobs), 1):
            rows.extend(r)
            conv.extend(c)
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)} images", flush=True)
    res = pd.DataFrame(rows)
    conv = pd.DataFrame(conv)

    metrics = ["AUC-Judd", "sAUC", "NSS", "CC", "SIM", "KL"]
    per_image = res.groupby(["model", "image", "collection"])[metrics].mean().reset_index()
    table = per_image.groupby("model")[metrics].agg(["mean", "std"])
    order = [o for o in ORDER if o in set(per_image.model)]
    print(per_image.groupby("model")[metrics].mean().loc[order].round(3).to_string())

    # --- figure 15: benchmark as small multiples
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, m in zip(axes.ravel(), metrics):
        means = per_image.groupby("model")[m].mean().loc[order]
        sems = per_image.groupby("model")[m].sem().loc[order] * 1.96
        ax.barh(range(len(order)), means, xerr=sems, color=[COLORS[o] for o in order],
                height=0.7, error_kw={"ecolor": INK2, "elinewidth": 1})
        for i, (v, e) in enumerate(zip(means, sems)):
            ax.text(v + e, i, f"  {v:.3f}", va="center", color=INK2, fontsize=8.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax in axes[:, 0] else [], fontsize=9)
        ax.invert_yaxis()
        better = "lower is better" if m == "KL" else "higher is better"
        ax.set_title(f"{m}  ({better})")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, (means + sems).max() * 1.25)
    groups = {"Human": "#2a78d6", "Deep saliency models": "#1baf7a", "Classic saliency models": "#6250d6",
              "Salient objects": "#eda100", "Centre bias": "#eb6834"}
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in groups.values()], list(groups),
               loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Saliency benchmark on the paintings (grey frame removed; split-half, mean ± 95% CI over paintings)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "15_benchmark_baselines.png")

    # --- figure 16: observer convergence
    conv_img = conv.groupby(["image", "collection", "k"])[["CC", "NSS", "AUC-Judd"]].mean().reset_index()
    n_img = conv_img.groupby("k").image.nunique()
    max_k = conv_img.groupby("image").k.max()
    conv_img = conv_img[conv_img.image.isin(max_k[max_k >= K_MAX].index)]
    ks = list(range(1, K_MAX + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    summary_conv = {}
    for ax, m in zip(axes, ["CC", "NSS", "AUC-Judd"]):
        g = conv_img[conv_img.k.isin(ks)].groupby("k")[m]
        mean, ci = g.mean(), g.sem() * 1.96
        ax.fill_between(mean.index, mean - ci, mean + ci, color="#cde2fb", lw=0)
        ax.plot(mean.index, mean, color="#2a78d6", marker="o", ms=5)
        ax.set_xlabel("Observers used to build the map")
        ax.set_title(m)
        ax.set_xticks(list(ks))
        summary_conv[m] = {int(k): round(v, 4) for k, v in mean.items()}
        top = mean.iloc[-1]
        summary_conv[m]["k_reaching_95pct_of_max"] = int(mean.index[np.argmax(mean.to_numpy() >= 0.95 * top)])
    fig.suptitle(f"How many observers are needed? Map from k observers vs {HELD_OUT} held-out observers "
                 f"(n = {conv_img.image.nunique()} paintings, mean ± 95% CI)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "16_observer_convergence.png")

    out = {
        "setup": {"evaluated_on": "painting only (grey frame removed), half resolution",
                  "sigma_px_full_res": round(ppd, 2), "splits_per_image": N_SPLITS,
                  "fixations_on_paintings": int(len(on)), "fixations_total": int(len(fix)),
                  "convergence_repeats": N_REPEATS_CONVERGENCE, "convergence_held_out": HELD_OUT,
                  "convergence_images": int(conv_img.image.nunique()),
                  "gaussian_centre_sd": "quarter of painting width / height",
                  "dataset_centre_bias": "other paintings' fixations in painting coordinates",
                  "sAUC_negatives": "20,000 fixations from other paintings, in painting coordinates",
                  "models": list(model_files)},
        "benchmark": {m: {k: {"mean": round(table.loc[m, (k, "mean")], 4), "sd": round(table.loc[m, (k, "std")], 4)}
                          for k in metrics} for m in order},
        "observer_convergence": summary_conv,
        "images_per_k": {int(k): int(v) for k, v in n_img.items()},
    }
    with open(os.path.join(OUT, "results_benchmark.json"), "w") as f:
        json.dump(out, f, indent=2)
    per_image[per_image.model == "Human (other half)"].drop(columns="model").round(4).to_csv(
        os.path.join(OUT, "per_image_human_agreement.csv"), index=False)
    per_image.round(4).to_csv(os.path.join(OUT, "per_image_benchmark.csv"), index=False)


if __name__ == "__main__":
    main()
