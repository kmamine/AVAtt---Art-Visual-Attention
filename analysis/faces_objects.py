"""Faces and objects: how much attention do they get, and how fast? (image-only detections)

Usage (from the repository root):  python analysis/faces_objects.py   (after composition_detect.py)
Writes analysis/composition/instances/<image>.npz (object masks), figure 30, results_faces_objects.json.

Objects: YOLO26-seg (COCO) + YOLOE-26 open-vocabulary masks, grouped into categories.
Faces: YOLO26-pose face keypoints + YOLOE-26 "face", drawn as discs (at least 0.5 deg radius).
For each painting and category: share of fixations inside vs the other-paintings baseline (log2
enrichment), area share, first-fixation latency, and share of viewers reaching it within 1 s and 2 s.
"""
import json
import os
from multiprocessing import Pool

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from analysis import INK, INK2, OUT, SURFACE, TP, plt, save
from paintfix import BASE_COLS, COMP, baseline, ci95, crop_image, load_all

MODELS = os.path.join(TP, "models")
CATEGORIES = {
    "face": {"face"},
    "person": {"person"},
    "hand": {"hand"},
    "animal": {"horse", "dog", "cow", "bird", "cat", "sheep", "elephant", "bear", "zebra", "giraffe"},
    "vessel / vehicle": {"boat", "ship", "car", "bicycle", "motorcycle", "train", "truck", "bus", "airplane"},
    "building": {"building", "church"},
    "tree / plant": {"tree", "flower", "potted plant"},
    "food / fruit": {"fruit", "apple", "orange", "banana", "bowl", "cup", "wine glass", "bottle"},
    "text": {"text", "book"},
    "sky": {"sky"},
    "water": {"water"},
    "mountain": {"mountain"},
}
LABEL2CAT = {l: c for c, ls in CATEGORIES.items() for l in ls}


def save_instance_masks(det):
    import torch
    from ultralytics import YOLO, YOLOE
    from composition_detect import YOLOE_PROMPTS, rasterize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = YOLO(os.path.join(MODELS, "yolo26l-seg.pt"))
    yoloe = YOLOE(os.path.join(MODELS, "yoloe-26l-seg.pt"))
    cwd = os.getcwd()
    os.chdir(MODELS)                     # YOLOE fetches its text encoder into the working directory
    yoloe.set_classes(YOLOE_PROMPTS)
    os.chdir(cwd)
    os.makedirs(os.path.join(COMP, "instances"), exist_ok=True)
    for name in sorted(det):
        path = os.path.join(COMP, "instances", name + ".npz")
        if os.path.exists(path):
            continue
        crop = crop_image(det, name)
        h, w = crop.shape[:2]
        masks, labels, confs = [], [], []
        for model in (seg, yoloe):
            r = model(crop, conf=0.2, verbose=False, device=device)[0]
            if r.boxes is None or r.masks is None:
                continue
            for i, (c, s) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                m = rasterize([r.masks.xy[i]], (h, w))
                if m.sum() >= 25:
                    masks.append(m)
                    labels.append(r.names[int(c)])
                    confs.append(s)
        np.savez_compressed(path, masks=np.array(masks, bool).reshape(-1, h, w), labels=np.array(labels),
                            confs=np.array(confs, np.float32))


def category_masks(det, name, ppd):
    d = det[name]
    w, h = d["crop_size"]
    z = np.load(os.path.join(COMP, "instances", name + ".npz"))
    cats = {}
    for m, lab in zip(z["masks"], z["labels"]):
        c = LABEL2CAT.get(str(lab))
        if c and c != "face":
            cats[c] = cats.get(c, np.zeros((h, w), bool)) | m
    face = np.zeros((h, w), np.uint8)
    for f in d["faces"]:
        cv2.circle(face, (int(round(f["x"])), int(round(f["y"]))), int(round(max(f["r"], ppd / 2))), 1, -1)
    if face.any():
        cats["face"] = face.astype(bool)
    return cats


def per_painting(job):
    name, det, obs, base, ppd = job
    cats = category_masks(det, name, ppd)
    on, bon = obs[obs.on_painting], base[base.on_painting]
    rows = []
    for c, m in cats.items():
        hh, ww = m.shape

        def inside(df):
            return m[np.clip((df.v * hh).astype(int), 0, hh - 1), np.clip((df.u * ww).astype(int), 0, ww - 1)]
        o, b = inside(on).mean(), inside(bon).mean()
        first_o = obs[obs.on_painting & inside(obs)].groupby(["image", "observer"]).t_s.min()
        first_b = base[base.on_painting & inside(base)].groupby(["image", "observer"]).t_s.min()
        n_o, n_b = obs.groupby(["image", "observer"]).ngroups, base.groupby(["image", "observer"]).ngroups
        rows.append({"image": name, "category": c, "area": m.mean(), "obs": o, "base": b,
                     "log2_enrichment": np.log2((o + 1e-3) / (b + 1e-3)),
                     "latency_obs": first_o.median() if len(first_o) else np.nan,
                     "latency_base": first_b.median() if len(first_b) else np.nan,
                     "reach1_obs": (first_o < 1).sum() / n_o, "reach1_base": (first_b < 1).sum() / n_b,
                     "reach2_obs": (first_o < 2).sum() / n_o, "reach2_base": (first_b < 2).sum() / n_b})
    return rows


def main():
    det, fix, ppd = load_all()
    save_instance_masks(det)
    names = sorted(det)
    jobs = [(n, det, fix.loc[fix.image == n, BASE_COLS], baseline(fix, n, seed=i), ppd) for i, n in enumerate(names)]
    with Pool(min(6, max(1, os.cpu_count() - 2))) as pool:
        rows = [r for rr in pool.map(per_painting, jobs) for r in rr]
    df = pd.DataFrame(rows)
    df.round(5).to_csv(os.path.join(OUT, "per_image_faces_objects.csv"), index=False)

    summ = []
    for c, g in df.groupby("category"):
        if len(g) < 5:
            continue
        summ.append({"category": c, "n_paintings": len(g), "area_share": g.area.mean(), "obs_share": g.obs.mean(),
                     "base_share": g.base.mean(), "enrich": g.log2_enrichment.mean(), "enrich_ci": ci95(g.log2_enrichment),
                     "p": stats.wilcoxon(g.obs, g.base).pvalue,
                     "latency_obs": g.latency_obs.median(), "latency_base": g.latency_base.median(),
                     "reach1_obs": g.reach1_obs.mean(), "reach1_base": g.reach1_base.mean(),
                     "reach2_obs": g.reach2_obs.mean(), "reach2_base": g.reach2_base.mean()})
    s = pd.DataFrame(summ).sort_values("enrich")
    from paintfix import bh
    s["q"] = bh(s.p.to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    colors = ["#2a78d6" if v > 0 else "#eb6834" for v in s.enrich]
    ax.barh(range(len(s)), s.enrich, xerr=s.enrich_ci, color=colors, height=0.65, error_kw={"ecolor": INK2})
    x_text = (s.enrich + s.enrich_ci).max() + 0.15
    for i, rrow in enumerate(s.itertuples()):
        ax.text(x_text, i, f"n={rrow.n_paintings}: {100 * rrow.obs_share:.1f}% of fixations on {100 * rrow.area_share:.1f}% of area"
                + ("*" if rrow.q < 0.05 else ""), va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.category)
    ax.set_xlim(min(0, (s.enrich - s.enrich_ci).min()) - 0.2, x_text + 3.2)
    ax.set_xlabel("log2 enrichment vs other-paintings baseline (* q < 0.05)")
    ax.set_title("Attention to faces and objects")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    s2 = s.sort_values("reach1_obs")
    y = np.arange(len(s2))
    ax.barh(y - 0.2, 100 * s2.reach1_obs, 0.38, color="#2a78d6", label="Observed")
    ax.barh(y + 0.2, 100 * s2.reach1_base, 0.38, color="#f4b393", label="Baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(s2.category)
    ax.set_xlabel("% of viewers fixating it within the first second")
    ax.set_title("How quickly it is reached")
    ax.legend()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "30_faces_objects.png")

    face = df[df.category == "face"]
    out = {"categories": json.loads(s.round(5).to_json(orient="records")),
           "faces": {"paintings_with_faces": int(len(face)), "n_faces": int(sum(len(d["faces"]) for d in det.values())),
                     "obs_share": float(face.obs.mean()), "base_share": float(face.base.mean()),
                     "area_share": float(face.area.mean()),
                     "latency_obs_s": float(face.latency_obs.median()), "latency_base_s": float(face.latency_base.median()),
                     "p_latency": float(stats.wilcoxon((face.latency_obs - face.latency_base).dropna()).pvalue)}}
    with open(os.path.join(OUT, "results_faces_objects.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
