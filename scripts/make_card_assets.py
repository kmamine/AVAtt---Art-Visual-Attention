"""Visuals for the dataset card and README (gallery, example panel, overlay video, GIF).

Usage:  python scripts/make_card_assets.py --data <AVAtt dir> --out <assets dir>
"""
import argparse
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import avatt
from avatt import geometry, maps

SCREAM = "avatt_0003"
SCANPATH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def overlay(img: np.ndarray, sal: np.ndarray, gamma=0.7, strength=0.85) -> np.ndarray:
    """Inferno heat over the painting, transparent where there is no attention."""
    a = (np.clip(sal, 0, 1) ** gamma)[..., None] * strength
    out = img.astype(float) / 255 * (1 - a) + cm.inferno(sal)[..., :3] * a
    return (out * 255).astype(np.uint8)


def font(size):
    for f in ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def gallery(ds, out, h=260, cols=4):
    """One painting per movement group (most observed), with its saliency overlay and a caption."""
    md = ds.metadata
    picks = []
    for g, sub in md.groupby("movement_group"):
        sub = sub[(sub.attribution_confidence == "high") & (sub.duplicate_of == "")]
        if sub.empty:
            sub = md[md.movement_group == g]
        picks.append((g, sub.sort_values("n_observers", ascending=False).index[0]))
    picks.sort()
    tiles = []
    for g, i in picks:
        img, sal = ds.image(i, crop=True), ds.saliency_map(i, crop=True)
        im = Image.fromarray(overlay(img, sal))
        im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
        tiles.append((g, md.loc[i], im))
    w = 340
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * w, rows * (h + 56)), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    for k, (g, r, im) in enumerate(tiles):
        x, y = (k % cols) * w, (k // cols) * (h + 56)
        if im.width > w - 12:
            im = im.resize((w - 12, int(im.height * (w - 12) / im.width)), Image.LANCZOS)
        canvas.paste(im, (x + (w - im.width) // 2, y + 6 + (h - im.height) // 2))
        d.text((x + 10, y + h + 10), g, fill=(11, 11, 11), font=font(17))
        sub = (r.artist or "Unattributed")[:34]
        d.text((x + 10, y + h + 32), sub, fill=(82, 81, 78), font=font(13))
    canvas.save(os.path.join(out, "gallery.jpg"), quality=88)


def example_panel(ds, out, image_id=SCREAM):
    img = ds.image(image_id, crop=True)
    x0, y0, _, _ = ds.painting_box(image_id)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    axes[0].imshow(img)
    axes[0].set_title("Painting")
    from scipy.ndimage import binary_dilation
    axes[1].imshow(binary_dilation(ds.fixation_map(image_id, crop=True), iterations=4), cmap="gray",
                   interpolation="nearest")
    axes[1].set_title(f"Fixations ({int(ds.metadata.loc[image_id, 'n_fixations'])})")
    axes[2].imshow(overlay(img, ds.saliency_map(image_id, crop=True)))
    axes[2].set_title("Saliency map (σ = 1°)")
    axes[3].imshow(img)
    for sp, c in zip(ds.scanpaths(image_id, max_time_ms=5000)[:3], SCANPATH_COLORS):
        axes[3].plot(sp.x - x0, sp.y - y0, "-o", color=c, lw=1.8, ms=6, mec="white", mew=0.8)
    axes[3].set_xlim(0, img.shape[1])
    axes[3].set_ylim(img.shape[0], 0)
    axes[3].set_title("Three scanpaths (5 s)")
    for ax in axes:
        ax.set_axis_off()
        ax.title.set_fontsize(14)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "example.jpg"), dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlay_video(ds, out, image_id=SCREAM, fps=30, scale=0.5):
    """Painting with attention accumulating over 5 s, as an H.264 mp4."""
    x0, y0, x1, y1 = ds.painting_box(image_id)
    img = ds.image(image_id, crop=True)
    hh, ww = (int((y1 - y0) * scale) // 2) * 2, (int((x1 - x0) * scale) // 2) * 2
    small = np.asarray(Image.fromarray(img).resize((ww, hh), Image.LANCZOS))
    f = ds.fixations([image_id], max_time_ms=5000)
    f["x"], f["y"] = ((f.x - x0) * scale).astype(int), ((f.y - y0) * scale).astype(int)
    path = os.path.join(out, "attention.mp4")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                             f"{ww}x{hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "23", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for k in range(1, 5 * fps + 1):
        m = maps.temporal_saliency_map(f, k * 1000 / fps, shape=(hh, ww), sigma=geometry.PX_PER_DEGREE * scale)
        fr = Image.fromarray(overlay(small, m))
        ImageDraw.Draw(fr).text((12, hh - 34), f"t = {k / fps:.1f} s", fill=(255, 255, 255), font=font(22))
        proc.stdin.write(np.asarray(fr).tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = avatt.load(a.data, download_if_missing=False)
    gallery(ds, a.out)
    example_panel(ds, a.out)
    overlay_video(ds, a.out)
    print({f: round(os.path.getsize(os.path.join(a.out, f)) / 1e6, 2) for f in sorted(os.listdir(a.out))})


if __name__ == "__main__":
    main()
