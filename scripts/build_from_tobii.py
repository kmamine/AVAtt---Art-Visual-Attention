"""Build the organised AVAtt dataset.

Usage: python -I build_avatt.py <Thesis_Dataset dir> <output dir>
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image

W, H = 1600, 900
FPS = 60
VIDEO_MS = 5000            # shortest presentation time across collections
DOWN = 4                   # temporal maps are accumulated at 1/4 resolution
MIN_RECORDING_MS = 10_000  # drops the single truncated recording
MIN_SCANPATH_MS = 4500     # first fixation onset to last fixation end
MIN_OBSERVERS = 10
COLLECTIONS = ["blois", "labO_tests_ET", "newDS_AVA"]
COLS = ["MediaName", "RecordingTimestamp", "GazeEventDuration", "RecordingDuration",
        "FixationPointX (MCSpx)", "FixationPointY (MCSpx)", "DistanceLeft", "DistanceRight",
        "GazePointLeftX (ADCSpx)", "GazePointLeftX (ADCSmm)"]


def clean_name(filename):
    stem = os.path.splitext(filename)[0]
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"\s+", "_", stem)
    return re.sub(r"[^A-Za-z0-9_.,()\-]", "", stem)


def eye_distance(df):
    left = df["DistanceLeft"].where(df["DistanceLeft"] > 0)
    right = df["DistanceRight"].where(df["DistanceRight"] > 0)
    return pd.concat([left, right], axis=1).mean(axis=1)


def load(root):
    """Return {image_name: (source_path, [observer fixation tables])} and px/mm."""
    images, px_per_mm = {}, []
    for coll in COLLECTIONS:
        available = set(os.listdir(os.path.join(root, coll, "images")))
        for rec in sorted(glob.glob(os.path.join(root, coll, "raw_scanpaths", "*.tsv"))):
            df = pd.read_csv(rec, sep="\t", usecols=COLS, low_memory=False)
            if df["RecordingDuration"].iloc[0] < MIN_RECORDING_MS:
                continue
            g = df.dropna(subset=["GazePointLeftX (ADCSpx)", "GazePointLeftX (ADCSmm)"])
            if len(g) > 10:
                px_per_mm.append(np.polyfit(g["GazePointLeftX (ADCSmm)"], g["GazePointLeftX (ADCSpx)"], 1)[0])
            df["distance_mm"] = eye_distance(df)
            run = (df["MediaName"] != df["MediaName"].shift()).cumsum()
            seen = set()
            for _, view in df.groupby(run, sort=True):
                media = view["MediaName"].iloc[0]
                if media not in available or media in seen:
                    continue  # interstimulus grey screens / first viewing only
                seen.add(media)
                x = view["FixationPointX (MCSpx)"]
                y = view["FixationPointY (MCSpx)"]
                view = view[(x >= 0) & (x < W) & (y >= 0) & (y < H)]
                if view.empty:
                    continue
                span = (view["RecordingTimestamp"].iloc[-1] + view["GazeEventDuration"].iloc[-1]
                        - view["RecordingTimestamp"].iloc[0])
                if span < MIN_SCANPATH_MS:
                    continue
                fix = pd.DataFrame({
                    "x": view["FixationPointX (MCSpx)"].astype(int),
                    "y": view["FixationPointY (MCSpx)"].astype(int),
                    "onset_ms": view["RecordingTimestamp"] - view["RecordingTimestamp"].iloc[0],
                    "duration_ms": view["GazeEventDuration"].astype(int),
                    "distance_mm": view["distance_mm"].round(1),
                })
                name = clean_name(media)
                src = os.path.join(root, coll, "images", media)
                images.setdefault(name, (src, []))[1].append(fix)
    images = {n: v for n, v in images.items() if len(v[1]) >= MIN_OBSERVERS}
    return images, float(np.mean(px_per_mm))


def write_video(path, frames_iter):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "20", "-preset", "medium", path],
        stdin=subprocess.PIPE)
    for frame in frames_iter:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def to_heatmap(m):
    peak = m.max()
    gray = np.uint8(255 * m / peak) if peak > 0 else np.zeros(m.shape, np.uint8)
    return gray


def build_image(job):
    name, src, observers, sigma, out = job
    shutil.copyfile(src, os.path.join(out, "images", name + ".jpg"))

    with open(os.path.join(out, "scanpaths", name + ".jsonl"), "w") as f:
        for i, fix in enumerate(observers, 1):
            fixations = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
                         for r in fix.astype(object).to_dict("records")]
            f.write(json.dumps({"observer": i, "fixations": fixations}) + "\n")

    allfix = pd.concat(observers, ignore_index=True)
    xs, ys = allfix["x"].to_numpy(), allfix["y"].to_numpy()

    fixmap = np.zeros((H, W), bool)
    fixmap[ys, xs] = True
    Image.fromarray(fixmap).save(os.path.join(out, "fixation_maps", name + ".png"))

    counts = np.zeros((H, W), np.float32)
    np.add.at(counts, (ys, xs), 1)
    sal = cv2.GaussianBlur(counts, (0, 0), sigma)
    cv2.imwrite(os.path.join(out, "saliency_heatmaps", name + ".png"), to_heatmap(sal))

    # Temporal maps: at time t each fixation contributes the time already spent on it.
    lx, ly = xs // DOWN, ys // DOWN
    onset = allfix["onset_ms"].to_numpy(float)
    dur = allfix["duration_ms"].to_numpy(float)

    def frames():
        for k in range(1, VIDEO_MS * FPS // 1000 + 1):
            t = k * 1000.0 / FPS
            w = np.clip(t - onset, 0, dur)
            acc = np.zeros((H // DOWN, W // DOWN), np.float32)
            np.add.at(acc, (ly, lx), w)
            small = cv2.GaussianBlur(acc, (0, 0), sigma / DOWN)
            full = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC).clip(min=0)
            yield cv2.applyColorMap(to_heatmap(full), cv2.COLORMAP_JET)

    write_video(os.path.join(out, "temporal_saliency_maps", name + ".mp4"), frames())
    return name, len(observers), len(allfix)


def main():
    root, out = sys.argv[1], sys.argv[2]
    for d in ["images", "scanpaths", "fixation_maps", "saliency_heatmaps", "temporal_saliency_maps"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)

    images, px_per_mm = load(root)
    distances = pd.concat([f["distance_mm"] for _, obs in images.values() for f in obs]).dropna()
    mean_dist = distances.mean()
    px_per_deg = 2 * mean_dist * np.tan(np.deg2rad(0.5)) * px_per_mm
    print(f"images={len(images)} px/mm={px_per_mm:.4f} screen_width_mm={1920 / px_per_mm:.1f} "
          f"mean_distance_mm={mean_dist:.1f} px_per_degree={px_per_deg:.2f}", flush=True)

    jobs = [(n, src, obs, px_per_deg, out) for n, (src, obs) in sorted(images.items())]
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for i, (name, n_obs, n_fix) in enumerate(pool.imap_unordered(build_image, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name}: {n_obs} observers, {n_fix} fixations", flush=True)


if __name__ == "__main__":
    main()
