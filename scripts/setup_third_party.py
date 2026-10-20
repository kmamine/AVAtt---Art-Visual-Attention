"""Fetch the third-party models used by the analysis pipeline into third_party/ (pinned commits).

Usage (from the repository root):  python scripts/setup_third_party.py [--skip-yolo]

Repositories                 commit     used for
  CACNet-Pytorch             2a287f1    composition classification (+ weights from the authors' Google Drive)
  GBVS_Pytorch               82371c6    GBVS and Itti-Koch saliency
  unisal                     7f8ca1a    UNISAL saliency (weights in repository)
  DeepGaze                   c7db17e    DeepGaze IIE saliency (+ MIT1003 centre bias)
  TranSalNet                 a2aba83    TranSalNet-Res saliency (+ weights from the authors' Google Drive)
Model files (third_party/models): YOLO26 / YOLOE-26 (Ultralytics), VICReg ResNet-50, DeepGaze centre bias.
BiRefNet, DINOv2/v3, MAE, CLIP, ViT, I-JEPA and torchvision models are fetched automatically on first use.
"""
import argparse
import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.environ.get("AVATT_THIRD_PARTY", os.path.join(ROOT, "third_party"))
MODELS = os.path.join(TP, "models")

REPOS = {
    "CACNet-Pytorch": ("https://github.com/bo-zhang-cs/CACNet-Pytorch.git", "2a287f19d2d5c21f0a0ac7d083f7fee3074c3dd9"),
    "GBVS_Pytorch": ("https://github.com/kmamine/GBVS_Pytorch.git", "82371c68b5e3538ea1014becefc845d0623633ac"),
    "unisal": ("https://github.com/rdroste/unisal.git", "7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9"),
    "DeepGaze": ("https://github.com/matthias-k/DeepGaze.git", "c7db17e2d1d7ea6468ffdee2cfaddf141095dcff"),
    "TranSalNet": ("https://github.com/LJOVO/TranSalNet.git", "a2aba83e3b8f54c47b712511bf4f515f236326ed"),
}
GDRIVE = {  # (file id, destination)
    "CACNet": ("1PK3FaId80M5yKywlMc2Uy6pHzQ_aqG4X", os.path.join(TP, "CACNet-Pytorch", "pretrained_model", "best-acc.pth")),
    "TranSalNet": ("14czAAQQcRLGeiddPOM6AaTJTieu6QiHy", os.path.join(TP, "TranSalNet", "pretrained_models", "TranSalNet_Res.pth")),
}
URLS = {
    os.path.join(MODELS, "vicreg_resnet50.pth"): "https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
    os.path.join(MODELS, "centerbias_mit1003.npy"):
        "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy",
    os.path.join(TP, "TranSalNet", "pretrained_models", "resnet50-0676ba61.pth"):
        "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}
YOLO = ["yolo26l-seg.pt", "yolo26l-pose.pt", "yolo26l-sem-ade20k.pt", "yolo26l-depth.pt", "yoloe-26l-seg.pt"]


def retry(fn, tries=5, wait=10):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # network hiccups
            if k == tries - 1:
                raise
            print(f"   retrying after error: {e}")
            time.sleep(wait)


def run(*cmd, cwd=None, check=True):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=check)


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}")
    retry(lambda: urllib.request.urlretrieve(url, dest + ".part"))
    os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-yolo", action="store_true", help="do not pre-download the Ultralytics weights")
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)
    for name, (url, sha) in REPOS.items():
        d = os.path.join(TP, name)
        if not os.path.isdir(os.path.join(d, ".git")):
            retry(lambda: run("git", "clone", "--quiet", url, d))
        if run("git", "cat-file", "-e", sha + "^{commit}", cwd=d, check=False).returncode != 0:
            retry(lambda: run("git", "fetch", "--quiet", "origin", sha, cwd=d))
        run("git", "checkout", "--quiet", sha, cwd=d)
    import gdown
    for name, (fid, dest) in GDRIVE.items():
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            retry(lambda: gdown.download(id=fid, output=dest, quiet=False))
    for dest, url in URLS.items():
        fetch(url, dest)
    if not a.skip_yolo:
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        from ultralytics import YOLO as Y, YOLOE
        cwd = os.getcwd()
        os.chdir(MODELS)                      # Ultralytics saves weights in the working directory
        try:
            for w in YOLO:
                if not os.path.exists(w):
                    (YOLOE if w.startswith("yoloe") else Y)(w)
            m = YOLOE("yoloe-26l-seg.pt")
            m.set_classes(["person"])          # fetches the MobileCLIP text encoder once
        finally:
            os.chdir(cwd)
    print("third-party models ready in", TP)


if __name__ == "__main__":
    main()
