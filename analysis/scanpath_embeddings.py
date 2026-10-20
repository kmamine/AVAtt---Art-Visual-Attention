"""Image-encoder scanpath similarity: the scanpath-nlp-metrics idea without the language model.

Usage (from the repository root):  python analysis/scanpath_embeddings.py   (after scanpath_similarity.py)
Writes analysis/similarity/scores_embed_<model>.csv (one row per pair of the shared design).

scanpath-nlp-metrics: fixation -> 96-px patch -> VLM description -> scanpath paragraph ->
ROUGE / BLEU / BM25 / BERTScore between paragraphs.
Here: fixation -> 96-px window -> image-encoder embedding (no text), then
  bertF1    greedy cosine matching of fixation embeddings, F1 of precision and recall (BERTScore analogue)
  dtw       dynamic time warping over (1 - cosine), normalised by path length (order-aware)
  meanpool  cosine between the mean embeddings of the two scanpaths (bag of what was looked at)
Each painting (grey frame removed) is encoded once per model, keeping its aspect ratio (long side 1600 px
for CNNs, 1120 px for ViTs); a fixation's embedding is the mean of the dense feature map over its window
(integral image), L2-normalised. Fixations that fell on the grey frame are left out of the sequences.
"""
import os
import sys

import cv2
import numba
import numpy as np
import pandas as pd
import torch

from analysis import DATA, OUT, TP

SIM = os.path.join(OUT, "similarity")
MODELS_DIR = os.path.join(TP, "models")
PATCH_PX = 96
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# name -> (family, description)
MODELS = {
    "vgg16": ("CNN (supervised)", "VGG-16, ImageNet-1k, conv5_3"),
    "resnet50": ("CNN (supervised)", "ResNet-50, ImageNet-1k (V2), layer4"),
    "convnext_tiny": ("CNN (supervised)", "ConvNeXt-T, ImageNet-1k, last stage"),
    "vit_b16_in21k": ("ViT (supervised)", "ViT-B/16 original (Dosovitskiy et al.), ImageNet-21k"),
    "clip_vit_b16": ("ViT (language-supervised)", "CLIP ViT-B/16 (OpenAI)"),
    "mae_vit_b16": ("Self-supervised", "MAE ViT-B/16"),
    "dinov2_vit_b14": ("Self-supervised", "DINOv2 ViT-B/14"),
    "dinov3_vit_b16": ("Self-supervised", "DINOv3 ViT-B/16"),
    "ijepa_vit_h14": ("Self-supervised", "I-JEPA ViT-H/14, ImageNet-1k"),
    "vicreg_resnet50": ("Self-supervised", "VICReg ResNet-50, layer4"),
}
TIMM_NAMES = {"convnext_tiny": "convnext_tiny.fb_in1k", "vit_b16_in21k": "vit_base_patch16_224.orig_in21k",
              "clip_vit_b16": "vit_base_patch16_clip_224.openai", "mae_vit_b16": "vit_base_patch16_224.mae",
              "dinov2_vit_b14": "vit_base_patch14_dinov2.lvd142m", "dinov3_vit_b16": "vit_base_patch16_dinov3.lvd1689m"}
IJEPA_REV = "f157467ea509bc356ff9f61fd3c0d840eec5e04e"


def fit(h, w, long_side, mult):
    """(width, height) with the given long side, both multiples of `mult`, aspect kept."""
    s = long_side / max(h, w)
    return max(mult, int(round(w * s / mult)) * mult), max(mult, int(round(h * s / mult)) * mult)


def to_tensor(rgb, size, mean, std):
    x = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))[None].to(DEVICE)


def load_encoder(name):
    """Return f(rgb 900x1600) -> feature map (C, h, w) float32 tensor on DEVICE."""
    import torchvision
    if name in ("vgg16", "resnet50", "vicreg_resnet50"):
        if name == "vgg16":
            net = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:30]
        else:
            m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if name == "resnet50" else None)
            if name == "vicreg_resnet50":
                sd = torch.load(os.path.join(MODELS_DIR, "vicreg_resnet50.pth"), map_location="cpu", weights_only=True)
                m.load_state_dict(sd, strict=False)
            net = torch.nn.Sequential(*list(m.children())[:-2])
        net = net.to(DEVICE).eval()

        def f(rgb):
            with torch.no_grad(), torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                return net(to_tensor(rgb, fit(*rgb.shape[:2], 1600, 32), *IMNET))[0].float()
        return f

    if name == "ijepa_vit_h14":
        from transformers import IJepaModel
        m = IJepaModel.from_pretrained("facebook/ijepa_vith14_1k", revision=IJEPA_REV, torch_dtype=torch.float16).to(DEVICE).eval()
        p = 14

        def f(rgb):
            size = fit(*rgb.shape[:2], 1120, p)
            with torch.no_grad():
                out = m(pixel_values=to_tensor(rgb, size, *IMNET).half(), interpolate_pos_encoding=True)
            tok = out.last_hidden_state[0].float()                     # (N, C), no CLS token
            gh, gw = size[1] // p, size[0] // p
            return tok[-gh * gw:].T.reshape(-1, gh, gw)
        return f

    import timm
    m = timm.create_model(TIMM_NAMES[name], pretrained=True, **({} if name == "convnext_tiny" else {"dynamic_img_size": True}))
    m = m.to(DEVICE).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    mean, std = cfg["mean"], cfg["std"]
    if name == "convnext_tiny":
        def f(rgb):
            with torch.no_grad(), torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                return m.forward_features(to_tensor(rgb, fit(*rgb.shape[:2], 1600, 32), mean, std))[0].float()
        return f
    p = m.patch_embed.patch_size[0]

    def f(rgb):
        size = fit(*rgb.shape[:2], 1120, p)
        with torch.no_grad(), torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            tok = m.forward_features(to_tensor(rgb, size, mean, std))[0].float()
        tok = tok[m.num_prefix_tokens:]
        gh, gw = size[1] // p, size[0] // p
        return tok.T.reshape(-1, gh, gw)
    return f


def pool_fixations(fmap, xs, ys, W, H):
    """Mean feature over each fixation's PATCH_PX window (cell centres inside), L2-normalised.
    xs, ys in painting-crop pixels; W, H = crop size."""
    C, h, w = fmap.shape
    if not len(xs):
        return np.zeros((0, C), np.float32)
    integ = torch.zeros((C, h + 1, w + 1), device=fmap.device)
    integ[:, 1:, 1:] = fmap.cumsum(1).cumsum(2)
    sx, sy = W / w, H / h
    half = PATCH_PX / 2
    x0 = np.clip(np.ceil((xs - half) / sx - 0.5), 0, w - 1).astype(int)
    x1 = np.clip(np.floor((xs + half) / sx - 0.5), 0, w - 1).astype(int)
    y0 = np.clip(np.ceil((ys - half) / sy - 0.5), 0, h - 1).astype(int)
    y1 = np.clip(np.floor((ys + half) / sy - 0.5), 0, h - 1).astype(int)
    cx = np.clip((xs / sx).astype(int), 0, w - 1)
    cy = np.clip((ys / sy).astype(int), 0, h - 1)
    empty = (x1 < x0) | (y1 < y0)
    x0, x1 = np.where(empty, cx, x0), np.where(empty, cx, x1)
    y0, y1 = np.where(empty, cy, y0), np.where(empty, cy, y1)
    t = lambda a: torch.as_tensor(a, device=fmap.device)
    s = integ[:, t(y1 + 1), t(x1 + 1)] - integ[:, t(y0), t(x1 + 1)] - integ[:, t(y1 + 1), t(x0)] + integ[:, t(y0), t(x0)]
    n = t((x1 - x0 + 1) * (y1 - y0 + 1)).float()
    e = (s / n).T
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy()


@numba.njit(cache=True)
def dtw_cost(D):
    n, m = D.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return acc[n, m] / (n + m)


def pair_scores(a, b):
    if not len(a) or not len(b):
        return {"bertF1": np.nan, "dtw": np.nan, "meanpool": np.nan}
    S = a @ b.T
    p, r = S.max(1).mean(), S.max(0).mean()
    ma, mb = a.mean(0), b.mean(0)
    return {"bertF1": float(2 * p * r / (p + r + 1e-12)),
            "dtw": float(dtw_cost((1 - S).astype(np.float64))),
            "meanpool": float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))}


def main(models=None):
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
    det = {i: {"crop_box": [int(r.painting_x0), int(r.painting_y0), int(r.painting_x1), int(r.painting_y1)]}
           for i, r in meta.iterrows()}
    sps = pd.read_csv(os.path.join(SIM, "scanpaths.csv"))
    pairs = pd.read_csv(os.path.join(SIM, "pairs.csv"))
    sp_by_image = {}
    for im, s in sps.groupby("image"):
        x0, y0, x1, y1 = det[im]["crop_box"]
        s = s[(s.x >= x0) & (s.x < x1) & (s.y >= y0) & (s.y < y1)]          # fixations on the painting only
        sp_by_image[im] = {sid: ((g.x - x0).to_numpy(float), (g.y - y0).to_numpy(float)) for sid, g in s.groupby("sid")}
    for name in (models or MODELS):
        path = os.path.join(SIM, f"scores_embed_{name}.csv")
        if os.path.exists(path):
            print(f"{name}: exists, skipped", flush=True)
            continue
        enc = load_encoder(name)
        rows = []
        for im, g in pairs.groupby("image"):
            x0, y0, x1, y1 = det[im]["crop_box"]
            rgb = cv2.cvtColor(cv2.imread(os.path.join(DATA, "images", im + ".jpg"))[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            fmap = enc(rgb)
            emb = {sid: pool_fixations(fmap, *xy, x1 - x0, y1 - y0) for sid, xy in sp_by_image[im].items()}
            empty = np.zeros((0, fmap.shape[0]), np.float32)
            for pid, a, b in zip(g.pair_id, g.sid_a, g.sid_b):
                rows.append({"pair_id": pid, **{f"{name}__{k}": v for k, v in
                                                pair_scores(emb.get(a, empty), emb.get(b, empty)).items()}})
        pd.DataFrame(rows).sort_values("pair_id").to_csv(path, index=False)
        print(f"{name}: done ({fmap.shape[0]} channels, {fmap.shape[1]}x{fmap.shape[2]} grid)", flush=True)
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
