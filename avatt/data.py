"""Download and read the AVAtt dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ID = "Amine-CV/AVAtt"
DEFAULT_DIR = Path(os.environ.get("AVATT_DATA", Path.home() / ".cache" / "avatt"))
FOLDERS = ("images", "scanpaths", "fixation_maps", "saliency_maps", "temporal_saliency_maps")


def download(local_dir: str | os.PathLike | None = None, repo_id: str = REPO_ID, revision: str | None = None,
             videos: bool = True, token: str | bool | None = None) -> Path:
    """Download the dataset files from the Hugging Face Hub and return the local folder.

    Set ``videos=False`` to skip the temporal saliency videos (about 85 MB).
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir or DEFAULT_DIR)
    patterns = ["metadata.csv", "images/*", "scanpaths/*", "fixation_maps/*", "saliency_maps/*"]
    if videos:
        patterns.append("temporal_saliency_maps/*")
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=str(local_dir),
                      allow_patterns=patterns, token=token)
    return local_dir


def load(root: str | os.PathLike | None = None, download_if_missing: bool = True, **kwargs) -> AVAtt:
    """Open the dataset at ``root`` (default: ``$AVATT_DATA`` or ``~/.cache/avatt``), downloading it if needed."""
    root = Path(root or DEFAULT_DIR)
    if not (root / "metadata.csv").exists():
        if not download_if_missing:
            raise FileNotFoundError(f"No AVAtt dataset at {root}")
        download(root, **kwargs)
    return AVAtt(root)


class AVAtt:
    """Read access to a local copy of the dataset.

    Coordinates are in canvas pixels (1600 x 900); ``painting_box`` gives the painting inside the grey canvas.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.metadata = pd.read_csv(self.root / "metadata.csv", keep_default_na=False).set_index("image_id")
        self.ids = list(self.metadata.index)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return (f"AVAtt(root='{self.root}', paintings={len(self)}, scanpaths={int(self.metadata.n_observers.sum())}, "
                f"fixations={int(self.metadata.n_fixations.sum())})")

    # ------------------------------------------------------------------ files
    def image(self, image_id: str, crop: bool = False) -> np.ndarray:
        """RGB stimulus (H x W x 3, uint8). ``crop=True`` returns the painting without the grey frame."""
        img = np.asarray(Image.open(self.root / "images" / f"{image_id}.jpg").convert("RGB"))
        return self._crop(img, image_id) if crop else img

    def fixation_map(self, image_id: str, crop: bool = False) -> np.ndarray:
        """Binary fixation map (H x W, bool)."""
        m = np.asarray(Image.open(self.root / "fixation_maps" / f"{image_id}.png")) > 0
        return self._crop(m, image_id) if crop else m

    def saliency_map(self, image_id: str, crop: bool = False) -> np.ndarray:
        """Saliency map smoothed at one degree of visual angle (H x W, float in [0, 1])."""
        m = np.asarray(Image.open(self.root / "saliency_maps" / f"{image_id}.png").convert("L"), np.float32) / 255
        return self._crop(m, image_id) if crop else m

    def video_path(self, image_id: str) -> Path:
        """Path of the temporal saliency video (60 fps, 5 s)."""
        return self.root / "temporal_saliency_maps" / f"{image_id}.mp4"

    def painting_box(self, image_id: str) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the painting inside the canvas."""
        r = self.metadata.loc[image_id]
        return int(r.painting_x0), int(r.painting_y0), int(r.painting_x1), int(r.painting_y1)

    def _crop(self, a: np.ndarray, image_id: str) -> np.ndarray:
        x0, y0, x1, y1 = self.painting_box(image_id)
        return a[y0:y1, x0:x1]

    # ------------------------------------------------------------------ gaze
    def scanpaths(self, image_id: str, max_time_ms: float | None = None) -> list[pd.DataFrame]:
        """One DataFrame per observer with columns x, y, onset_ms, duration_ms, distance_mm."""
        out = []
        with open(self.root / "scanpaths" / f"{image_id}.jsonl") as f:
            for line in f:
                sp = json.loads(line)
                df = pd.DataFrame(sp["fixations"])
                if max_time_ms is not None:
                    df = df[df.onset_ms < max_time_ms]
                df.attrs["observer"] = sp["observer"]
                out.append(df.reset_index(drop=True))
        return out

    def fixations(self, image_ids: list[str] | None = None, max_time_ms: float | None = None) -> pd.DataFrame:
        """All fixations as one long table with image_id and observer columns."""
        frames = []
        for i in image_ids or self.ids:
            for sp in self.scanpaths(i, max_time_ms):
                frames.append(sp.assign(image_id=i, observer=sp.attrs["observer"]))
        return pd.concat(frames, ignore_index=True)
