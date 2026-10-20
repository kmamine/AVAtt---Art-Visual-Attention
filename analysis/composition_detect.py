"""Image-only content and composition detection for every AVAtt painting (no fixation data used).

Usage (from the repository root):  python analysis/composition_detect.py
Writes results/composition/:
  detections.json          per painting: crop box, subjects, faces, objects, composition class
                           probabilities (CACNet, CLIP), lines, horizon, symmetry, visual-weight centroid
  sod/<image>.png          salient-object probability (BiRefNet), painting crop
  subjects/<image>.png     focal-subject label map (0 = none, 1..3 = subject rank), painting crop
  semantic/<image>.png     ADE20K class ids (YOLO26 semantic), painting crop
  depth/<image>.png        relative depth (YOLO26 depth, 255 = near), painting crop

All models run on the painting crop only (the grey frame is removed first).
"""
import gc
import json
import os
import sys

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

import cv2
import numpy as np
import torch
from PIL import Image

from analysis import DATA, OUT, TP, painting_mask

ROOT = os.path.dirname(OUT)
COMP = os.path.join(OUT, "composition")
MODELS = os.path.join(TP, "models")
CACNET = os.path.join(TP, "CACNet-Pytorch")
BIREFNET_REV = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPOSITION_CLASSES = ["rule of thirds", "vertical", "horizontal", "diagonal", "curved",
                       "triangle", "center", "symmetric", "pattern"]   # KU-PCP order (CACNet output order)
CLIP_PROMPTS = {
    "rule of thirds": ["a painting composed with the rule of thirds", "a painting whose main subject is placed off-centre on a thirds line"],
    "vertical": ["a painting composed of strong vertical lines", "a painting dominated by vertical elements"],
    "horizontal": ["a painting composed of strong horizontal lines", "a painting with a dominant horizon line"],
    "diagonal": ["a painting with a diagonal composition", "a painting composed along a strong diagonal line"],
    "curved": ["a painting with a curved composition", "a painting composed along sweeping curved lines"],
    "triangle": ["a painting with a triangular composition", "a painting with a pyramidal arrangement of figures"],
    "center": ["a painting with a centred composition", "a painting whose main subject is in the centre"],
    "symmetric": ["a symmetrical painting", "a painting with mirror symmetry"],
    "pattern": ["a painting made of a repeating pattern", "an abstract painting with repeated motifs"],
}
YOLOE_PROMPTS = ["face", "person", "hand", "horse", "dog", "cow", "bird", "boat", "ship", "building",
                 "church", "tree", "flower", "fruit", "text", "sky", "water", "mountain"]
STUFF = {"sky", "water", "mountain", "tree", "building", "church"}   # never focal subjects on their own
FACE_KPTS = [0, 1, 2, 3, 4]   # COCO nose, eyes, ears


def crops():
    """Yield (name, None, crop box (x0, y0, x1, y1), crop BGR, painting mask crop)."""
    for f in sorted(os.listdir(os.path.join(DATA, "images"))):
        name = f[:-4]
        bgr = cv2.imread(os.path.join(DATA, "images", f))
        paint = painting_mask(bgr)
        rows, cols = np.where(paint.any(1))[0], np.where(paint.any(0))[0]
        x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        yield name, None, (x0, y0, x1, y1), bgr[y0:y1, x0:x1].copy(), paint[y0:y1, x0:x1]


def save_png(folder, name, arr):
    os.makedirs(os.path.join(COMP, folder), exist_ok=True)
    cv2.imwrite(os.path.join(COMP, folder, name + ".png"), arr)


def free():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- pass 1: salient objects (BiRefNet)
def pass_sod(items):
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True, revision=BIREFNET_REV).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for name, _, _, crop, _ in items:
        x = tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))[None].to(DEVICE)
        with torch.no_grad():
            p = model(x.half() if DEVICE == "cuda" else x)[-1].sigmoid().float().cpu()[0, 0].numpy()
        p = cv2.resize(p, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        save_png("sod", name, np.uint8(np.clip(p, 0, 1) * 255))
    del model
    free()


# ---------------------------------------------------------------- pass 2: YOLO26 family
def rasterize(polys, shape):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def pass_yolo(items):
    from ultralytics import YOLO, YOLOE
    seg = YOLO(os.path.join(MODELS, "yolo26l-seg.pt"))
    pose = YOLO(os.path.join(MODELS, "yolo26l-pose.pt"))
    sem = YOLO(os.path.join(MODELS, "yolo26l-sem-ade20k.pt"))
    depth = YOLO(os.path.join(MODELS, "yolo26l-depth.pt"))
    yoloe = YOLOE(os.path.join(MODELS, "yoloe-26l-seg.pt"))
    cwd = os.getcwd()
    os.chdir(MODELS)                     # YOLOE fetches its text encoder into the working directory
    yoloe.set_classes(YOLOE_PROMPTS)
    os.chdir(cwd)
    out = {}
    for name, _, box, crop, _ in items:
        h, w = crop.shape[:2]
        rec = {"instances": [], "faces": []}

        for model, source in [(seg, "yolo26-seg"), (yoloe, "yoloe-26")]:
            r = model(crop, conf=0.2, verbose=False, device=DEVICE)[0]
            if r.boxes is None or r.masks is None:
                continue
            for i, (c, s, b) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())):
                mask = rasterize([r.masks.xy[i]], (h, w))
                if mask.sum() < 25:
                    continue
                rec["instances"].append({"source": source, "label": r.names[int(c)], "conf": round(s, 3),
                                         "box": [round(v, 1) for v in b], "_mask": mask})

        r = pose(crop, conf=0.2, verbose=False, device=DEVICE)[0]
        if r.keypoints is not None and r.keypoints.conf is not None:
            for kp, kc, pb in zip(r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy(), r.boxes.xyxy.tolist()):
                vis = [k for k in FACE_KPTS if kc[k] > 0.4]
                if sum(k in vis for k in (0, 1, 2)) >= 2:
                    pts = kp[vis]
                    spread = float(np.ptp(pts[:, 0])) if len(pts) > 1 else 0.0
                    radius = max(spread * 0.9, 0.08 * (pb[3] - pb[1]), 6.0)
                    rec["faces"].append({"source": "yolo26-pose", "x": float(pts[:, 0].mean()),
                                         "y": float(pts[:, 1].mean()), "r": radius,
                                         "conf": round(float(kc[vis].mean()), 3)})
        for inst in rec["instances"]:
            if inst["label"] == "face":
                x0, y0, x1, y1 = inst["box"]
                f = {"source": "yoloe-26", "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                     "r": max(x1 - x0, y1 - y0) / 2, "conf": inst["conf"]}
                if all(np.hypot(f["x"] - g["x"], f["y"] - g["y"]) > max(f["r"], g["r"]) for g in rec["faces"]):
                    rec["faces"].append(f)

        r = sem(crop, verbose=False, device=DEVICE)[0]
        sm = getattr(r, "semantic_mask", None)
        if sm is not None:
            sm = sm.data if hasattr(sm, "data") else sm
            sm = sm.cpu().numpy() if hasattr(sm, "cpu") else np.asarray(sm)
            sm = np.squeeze(sm).astype(np.uint8)
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
            save_png("semantic", name, sm)
            rec["semantic_names"] = {int(k): v for k, v in r.names.items()}

        r = depth(crop, verbose=False, device=DEVICE)[0]
        dm = getattr(r, "depth", None)
        if dm is not None:
            dm = dm.data if hasattr(dm, "data") else dm
            dm = np.squeeze(dm.cpu().numpy() if hasattr(dm, "cpu") else np.asarray(dm)).astype(np.float32)
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
            dm = (dm - dm.min()) / (np.ptp(dm) + 1e-9)
            save_png("depth", name, np.uint8(255 * dm))
        out[name] = rec
    del seg, pose, sem, depth, yoloe
    free()
    return out


# ---------------------------------------------------------------- pass 3: composition classifiers
def pass_classifiers(items):
    import open_clip
    sys.path.insert(0, CACNET)
    from CACNet import ComClassifier
    from torchvision import transforms

    cac = ComClassifier(loadweights=False)
    cac.load_state_dict(torch.load(os.path.join(CACNET, "pretrained_model", "best-acc.pth"),
                                   map_location="cpu", weights_only=True))
    cac = cac.to(DEVICE).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEVICE)
    tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        text = []
        for c in COMPOSITION_CLASSES:
            e = clip.encode_text(tok(CLIP_PROMPTS[c]).to(DEVICE)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            text.append(e.mean(0))
        text = torch.stack(text)
        text = text / text.norm(dim=-1, keepdim=True)

    out = {}
    for name, _, _, crop, _ in items:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            logits, _ = cac(tf(pil)[None].to(DEVICE))
            p_cac = torch.softmax(logits, 1)[0].cpu().numpy()
            im = clip.encode_image(preprocess(pil)[None].to(DEVICE)).float()
            im = im / im.norm(dim=-1, keepdim=True)
            p_clip = torch.softmax(100 * im @ text.T, 1)[0].cpu().numpy()
        out[name] = {"cacnet": {c: round(float(v), 4) for c, v in zip(COMPOSITION_CLASSES, p_cac)},
                     "clip": {c: round(float(v), 4) for c, v in zip(COMPOSITION_CLASSES, p_clip)}}
    del cac, clip
    free()
    return out


# ---------------------------------------------------------------- pass 4: geometry from pixels (CPU)
def lines_and_symmetry(crop, sod):
    h, w = crop.shape[:2]
    scale = 800 / max(h, w)
    gray = cv2.cvtColor(cv2.resize(crop, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(gray)[0]
    diag = np.hypot(*gray.shape)
    lines, orient = [], {"horizontal": 0.0, "vertical": 0.0, "diagonal": 0.0, "other": 0.0}
    if segs is not None:
        for x0, y0, x1, y1 in segs[:, 0]:
            length = np.hypot(x1 - x0, y1 - y0)
            if length < 0.04 * diag:
                continue
            ang = np.degrees(np.arctan2(-(y1 - y0), x1 - x0)) % 180
            a = min(ang, 180 - ang)                       # 0 = horizontal, 90 = vertical
            key = "horizontal" if a < 15 else "vertical" if a > 75 else "diagonal" if 30 <= a <= 60 else "other"
            orient[key] += float(length)
            lines.append([float(x0 / scale), float(y0 / scale), float(x1 / scale), float(y1 / scale),
                          round(float(length / diag), 4)])
    total = sum(orient.values()) or 1.0
    orient = {k: round(v / total, 4) for k, v in orient.items()}
    lines = sorted(lines, key=lambda l: -l[4])[:40]

    small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (128, int(128 * h / w))).astype(float)
    mirror = float(np.corrcoef(small.ravel(), small[:, ::-1].ravel())[0, 1])
    s = sod > 0.5
    sod_sym = float((s & s[:, ::-1]).sum() / ((s | s[:, ::-1]).sum() + 1e-9))
    return lines, orient, round(mirror, 4), round(sod_sym, 4)


def horizon(sem_map, names):
    sky_ids = [k for k, v in names.items() if v == "sky"]
    if sem_map is None or not sky_ids:
        return None
    sky = np.isin(sem_map, sky_ids)
    if sky.mean() < 0.05:
        return None
    h = sky.shape[0]
    cols = [np.max(np.where(sky[:, j])[0]) for j in range(sky.shape[1]) if sky[:, j].any()]
    return round(float(np.median(cols)) / h, 4) if cols else None


def focal_subjects(instances, sod, name):
    """Rank instances by the share of salient-object mass they cover; keep up to 3 focal subjects."""
    total = sod.sum() + 1e-9
    cands = []
    for inst in instances:
        if inst["label"] in STUFF:
            continue
        share = float((sod * inst["_mask"]).sum() / total)
        cands.append((share, inst))
    cands.sort(key=lambda t: -t[0])
    chosen = []
    for share, inst in cands:
        if share < 0.10 or len(chosen) == 3:
            break
        if any((inst["_mask"] & c["_mask"]).sum() / ((inst["_mask"] | c["_mask"]).sum() + 1e-9) > 0.5 for c in chosen):
            continue
        inst = dict(inst, salience_share=round(share, 4))
        chosen.append(inst)
    if not chosen:   # nothing detected: fall back to salient-object components
        n, lab, stats, _ = cv2.connectedComponentsWithStats(np.uint8(sod > 0.5))
        comps = sorted(range(1, n), key=lambda i: -(sod * (lab == i)).sum())
        for i in comps[:3]:
            mask = lab == i
            share = float((sod * mask).sum() / total)
            if share < 0.10 or mask.mean() < 0.01:
                continue
            x, y, bw, bh, _ = stats[i]
            chosen.append({"source": "birefnet", "label": "salient region", "conf": None,
                           "box": [float(x), float(y), float(x + bw), float(y + bh)], "_mask": mask,
                           "salience_share": round(share, 4)})
    label_map = np.zeros(sod.shape, np.uint8)
    for rank, c in enumerate(chosen, 1):
        label_map[c["_mask"] & (label_map == 0)] = rank
    save_png("subjects", name, label_map)
    out = []
    for c in chosen:
        ys, xs = np.nonzero(c["_mask"])
        out.append({k: v for k, v in c.items() if k != "_mask"} |
                   {"cx": float(xs.mean()), "cy": float(ys.mean()), "area_frac": round(float(c["_mask"].mean()), 4)})
    return out


def main():
    os.makedirs(COMP, exist_ok=True)
    items = list(crops())
    print(f"{len(items)} paintings, device = {DEVICE}", flush=True)
    pass_sod(items)
    print("salient objects done", flush=True)
    yolo = pass_yolo(items)
    print("YOLO26 done", flush=True)
    cls = pass_classifiers(items)
    print("composition classifiers done", flush=True)

    det = {}
    for name, _, box, crop, paint in items:
        h, w = crop.shape[:2]
        sod = cv2.imread(os.path.join(COMP, "sod", name + ".png"), 0).astype(np.float32) / 255
        sem_path = os.path.join(COMP, "semantic", name + ".png")
        sem_map = cv2.imread(sem_path, 0) if os.path.exists(sem_path) else None
        lines, orient, mirror, sod_sym = lines_and_symmetry(crop, sod)
        ys, xs = np.mgrid[0:h, 0:w]
        wsum = sod.sum() + 1e-9
        rec = yolo[name]
        det[name] = {
            "crop_box": list(box), "crop_size": [w, h], "aspect_ratio": round(w / h, 4),
            "painting_area_frac": round(float(paint.mean() * (w * h) / (1600 * 900)), 4),
            "subjects": focal_subjects(rec["instances"], sod, name),
            "faces": rec["faces"],
            "objects": [{k: v for k, v in i.items() if k != "_mask"} | {"area_frac": round(float(i["_mask"].mean()), 5)}
                        for i in rec["instances"]],
            "salient_area_frac": round(float((sod > 0.5).mean()), 4),
            "visual_weight_centroid": [round(float((sod * xs).sum() / wsum / w), 4),
                                       round(float((sod * ys).sum() / wsum / h), 4)],
            "composition_probs": cls[name],
            "line_orientation_share": orient, "lines": lines,
            "mirror_symmetry": mirror, "salient_symmetry": sod_sym,
            "horizon_v": horizon(sem_map, rec.get("semantic_names", {})),
            "semantic_names": rec.get("semantic_names", {}),
        }
    with open(os.path.join(COMP, "detections.json"), "w") as f:
        json.dump(det, f)
    n_faces = sum(len(d["faces"]) for d in det.values())
    n_subj = sum(len(d["subjects"]) for d in det.values())
    print(f"done: {n_subj} focal subjects, {n_faces} faces", flush=True)


if __name__ == "__main__":
    main()
