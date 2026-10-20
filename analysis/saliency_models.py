"""Predict saliency maps for every AVAtt frame with classic and lightweight deep models.

Usage (from the repository root):  python analysis/saliency_models.py [model ...]
Writes analysis/saliency_models/<model>.npz : {image: float16 map of the painting crop at half resolution}.
Scored by benchmark.py (on the painting area only) together with the human and centre-bias baselines.

Every model sees the painting only: the grey frame around it is cropped away first (crop box from
composition_detect.py). Itti-Koch runs with the crop's long side resized to 800 px (about 60x faster).
  gbvs              GBVS, Harel et al. 2006 (kmamine/GBVS_Pytorch, default DIO channels)
  ittikoch          Itti, Koch & Niebur 1998 (same package, make_ittikoch_params)
  spectral_residual Hou & Zhang 2007 (64-px width, 3x3 average, Gaussian blur)
  unisal            UNISAL, Droste et al. 2020 (MobileNetV2, SALICON domain, static mode)
  deepgaze_iie      DeepGaze IIE, Linardos et al. 2021, with the MIT1003 centre bias it was built for
  transalnet_res    TranSalNet-Res, Lou et al. 2022 (SALICON weights)
  salient_objects   BiRefNet salient-object map from composition_detect.py (0 outside the painting)
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from analysis import DATA, OUT, TP

SAL = os.path.join(OUT, "saliency_models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALL = ["gbvs", "ittikoch", "spectral_residual", "unisal", "deepgaze_iie", "transalnet_res", "salient_objects"]


_META = __import__("pandas").read_csv(os.path.join(DATA, "metadata.csv"), keep_default_na=False).set_index("image_id")
DET = {i: {"crop_box": [int(r.painting_x0), int(r.painting_y0), int(r.painting_x1), int(r.painting_y1)]}
       for i, r in _META.iterrows()}


def images():
    """Yield (name, painting crop BGR)."""
    for f in sorted(os.listdir(os.path.join(DATA, "images"))):
        x0, y0, x1, y1 = DET[f[:-4]]["crop_box"]
        yield f[:-4], cv2.imread(os.path.join(DATA, "images", f))[y0:y1, x0:x1]


def half_size(name):
    x0, y0, x1, y1 = DET[name]["crop_box"]
    return max((x1 - x0) // 2, 1), max((y1 - y0) // 2, 1)


def half(m, name):
    m = cv2.resize(np.asarray(m, np.float32), half_size(name), interpolation=cv2.INTER_AREA)
    m = m - m.min()
    return (m / (m.max() + 1e-12)).astype(np.float16)


# ---------------------------------------------------------------- models
def gbvs_family(kind):
    sys.path.insert(0, os.path.join(TP, "GBVS_Pytorch"))
    from gbvs_torch import GBVSParams, gbvs, make_ittikoch_params

    def f(bgr):
        if kind == "ittikoch":
            s = 800 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
        t = torch.from_numpy(img)
        p = GBVSParams() if kind == "gbvs" else make_ittikoch_params(img.shape)
        p.exact_conv, p.matvec_mode = False, "blas"     # the package's documented fast path
        out, _ = gbvs(t, p)
        m = out["master_map_resized"]
        return m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    return f


def spectral_residual():
    def f(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (64, int(round(64 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
        F = np.fft.fft2(g)
        logamp, phase = np.log(np.abs(F) + 1e-9), np.angle(F)
        resid = logamp - cv2.blur(logamp, (3, 3))
        s = np.abs(np.fft.ifft2(np.exp(resid + 1j * phase))) ** 2
        return cv2.GaussianBlur(s.astype(np.float32), (0, 0), 3)
    return f


def unisal():
    sys.path.insert(0, os.path.join(TP, "unisal"))
    from unisal.model import UNISAL
    cfg = json.load(open(os.path.join(TP, "unisal", "training_runs", "pretrained_unisal", "UNISAL.json")))
    cfg["cnn_cfg"]["pretrained"] = False
    cfg.pop("verbose", None)
    model = UNISAL(**cfg)
    model.load_state_dict(torch.load(os.path.join(TP, "unisal", "training_runs", "pretrained_unisal", "weights_best.pth"),
                                     map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval()
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    def out_size(h, w):            # UNISAL's eval size rule (MIT data): 100-120 cells of 32 px, closest aspect
        ar = h / w
        best = max(((n1, n2) for n1 in range(7, 14) for n2 in range(7, 14) if 100 <= n1 * n2 <= 120),
                   key=lambda s: min(ar, s[0] / s[1]) / max(ar, s[0] / s[1]))
        return best[0] * 32, best[1] * 32

    def f(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = out_size(*rgb.shape[:2])
        x = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
        x = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1))[None, None].to(DEVICE)
        with torch.no_grad():
            pred = model(x, target_size=(rgb.shape[0] // 2, rgb.shape[1] // 2), source="SALICON", static=True)
        return pred[0, 0, 0].exp().cpu().numpy()
    return f


def deepgaze_iie():
    sys.path.insert(0, os.path.join(TP, "DeepGaze"))
    from scipy.ndimage import zoom
    from scipy.special import logsumexp
    import torchvision
    import deepgaze_pytorch
    # DeepGaze fetches DenseNet-201 / ResNeXt-50 via torch.hub('pytorch/vision:v0.6.0'), which downloads and runs
    # an old torchvision snapshot. Serve the same ImageNet weights from the installed torchvision instead.
    hub_load = torch.hub.load

    def local_hub(repo, name, pretrained=False, **kw):
        if repo == "pytorch/vision:v0.6.0":
            return getattr(torchvision.models, name)(weights="IMAGENET1K_V1" if pretrained else None)
        return hub_load(repo, name, pretrained=pretrained, **kw)
    torch.hub.load = local_hub
    try:
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE).eval()
    finally:
        torch.hub.load = hub_load
    cb0 = np.load(os.path.join(TP, "models", "centerbias_mit1003.npy"), allow_pickle=False)

    def f(bgr):
        h, w = bgr.shape[0] // 2, bgr.shape[1] // 2
        cb = zoom(cb0, (h / cb0.shape[0], w / cb0.shape[1]), order=0, mode="nearest")
        cb = cb - logsumexp(cb)
        cb_t = torch.tensor(cb[None], dtype=torch.float32, device=DEVICE)
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logd = model(x, cb_t)
        return logd[0, 0].exp().cpu().numpy()
    return f


def transalnet_res():
    root = os.path.join(TP, "TranSalNet")
    sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)                 # the model loads 'pretrained_models/...' relative to its folder
    try:
        from TranSalNet_Res import TranSalNet
        from utils.data_process import postprocess_img
        model = TranSalNet()
        model.load_state_dict(torch.load(os.path.join(root, "pretrained_models", "TranSalNet_Res.pth"),
                                         map_location="cpu", weights_only=True))
    finally:
        os.chdir(cwd)
    model = model.to(DEVICE).eval()

    R, C = 288, 384

    def geometry(h, w):            # same letterboxing as utils.data_process.preprocess_img / postprocess_img
        if h / R > w / C:
            nc = min((w * R) // h, C)
            return (slice(None), slice((C - nc) // 2, (C - nc) // 2 + nc)), (nc, R)
        nr = min((h * C) // w, R)
        return (slice((R - nr) // 2, (R - nr) // 2 + nr), slice(None)), (C, nr)

    def f(bgr):
        region, size = geometry(*bgr.shape[:2])
        canvas = np.ones((R, C, 3), np.uint8)
        canvas[region] = cv2.resize(bgr, size)
        x = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255).to(DEVICE)
        with torch.no_grad():
            p = model(x)[0, 0].cpu().numpy()
        return p[region]               # letterbox removed; resized by half()
    return f


def salient_objects():
    def f(bgr, name):
        return cv2.imread(os.path.join(OUT, "composition", "sod", name + ".png"), 0).astype(np.float32) / 255
    return f


BUILDERS = {"gbvs": lambda: gbvs_family("gbvs"), "ittikoch": lambda: gbvs_family("ittikoch"),
            "spectral_residual": spectral_residual, "unisal": unisal, "deepgaze_iie": deepgaze_iie,
            "transalnet_res": transalnet_res, "salient_objects": salient_objects}


def main(models):
    os.makedirs(SAL, exist_ok=True)
    for name in models:
        path = os.path.join(SAL, name + ".npz")
        if os.path.exists(path):
            print(f"{name}: exists, skipped", flush=True)
            continue
        f = BUILDERS[name]()
        maps = {}
        for i, (img_name, bgr) in enumerate(images()):
            m = f(bgr, img_name) if name == "salient_objects" else f(bgr)
            maps[img_name] = half(m, img_name)
            if i % 50 == 0:
                print(f"  {name}: {i}", flush=True)
        np.savez_compressed(path, **maps)
        print(f"{name}: done", flush=True)
        del f
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
