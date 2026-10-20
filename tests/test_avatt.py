"""Tests for the avatt package. Fast tests use a tiny synthetic dataset; set AVATT_DATA to also test a real copy."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import avatt
from avatt import geometry, maps, metrics


# ---------------------------------------------------------------- synthetic dataset
@pytest.fixture()
def tiny(tmp_path):
    for d in ("images", "scanpaths", "fixation_maps", "saliency_maps", "temporal_saliency_maps"):
        (tmp_path / d).mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in (1, 2):
        iid = f"avatt_{i:04d}"
        Image.fromarray(np.full((900, 1600, 3), 170, np.uint8)).save(tmp_path / "images" / f"{iid}.jpg")
        sps = []
        for o in range(3):
            n = 10
            fx = [{"x": int(800 + rng.normal(0, 50)), "y": int(450 + rng.normal(0, 50)), "onset_ms": 400 * k,
                   "duration_ms": 300, "distance_mm": 570.0} for k in range(n)]
            sps.append({"observer": o + 1, "fixations": fx})
        with open(tmp_path / "scanpaths" / f"{iid}.jsonl", "w") as f:
            f.writelines(json.dumps(sp) + "\n" for sp in sps)
        fixdf = [pd.DataFrame(sp["fixations"]) for sp in sps]
        Image.fromarray(maps.fixation_map(fixdf)).save(tmp_path / "fixation_maps" / f"{iid}.png")
        Image.fromarray((maps.saliency_map(fixdf) * 255).astype(np.uint8)).save(tmp_path / "saliency_maps" / f"{iid}.png")
        rows.append({"image_id": iid, "title": "", "artist": "", "n_observers": 3, "n_fixations": 30,
                     "painting_x0": 200, "painting_y0": 0, "painting_x1": 1400, "painting_y1": 900})
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path


def test_load_and_read(tiny):
    ds = avatt.load(tiny, download_if_missing=False)
    assert len(ds) == 2 and ds.ids == ["avatt_0001", "avatt_0002"]
    assert "paintings=2" in repr(ds)
    sps = ds.scanpaths("avatt_0001")
    assert len(sps) == 3 and list(sps[0].columns) == ["x", "y", "onset_ms", "duration_ms", "distance_mm"]
    assert sps[0].attrs["observer"] == 1
    assert len(ds.scanpaths("avatt_0001", max_time_ms=1000)[0]) == 3
    assert ds.image("avatt_0001").shape == (900, 1600, 3)
    assert ds.image("avatt_0001", crop=True).shape == (900, 1200, 3)
    assert ds.fixation_map("avatt_0001").dtype == bool
    assert 0 <= ds.saliency_map("avatt_0001").min() and ds.saliency_map("avatt_0001").max() <= 1
    allfix = ds.fixations()
    assert len(allfix) == 60 and {"image_id", "observer"} <= set(allfix.columns)


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        avatt.load(tmp_path / "nowhere", download_if_missing=False)


# ---------------------------------------------------------------- geometry and maps
def test_geometry():
    assert abs(geometry.PX_PER_DEGREE - 35.87) < 0.01
    assert abs(geometry.pixels_to_degrees(geometry.PX_PER_DEGREE) - 1.0) < 0.01


def test_maps():
    f = pd.DataFrame({"x": [10, 20, -5], "y": [5, 5, 5], "onset_ms": [0, 300, 600], "duration_ms": [300, 300, 300]})
    fm = maps.fixation_map(f, shape=(50, 50))
    assert fm.sum() == 2                                   # out-of-bounds fixation ignored
    sm = maps.saliency_map(f, shape=(50, 50), sigma=3)
    assert sm.max() == pytest.approx(1.0) and sm[5, 15] > sm[40, 40]
    early = maps.temporal_saliency_map(f, t_ms=150, shape=(50, 50), sigma=1, normalize=False)
    late = maps.temporal_saliency_map(f, t_ms=600, shape=(50, 50), sigma=1, normalize=False)
    assert early[5, 20] < 1e-9 and late[5, 20] > 0 and late.sum() > early.sum()


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_and_chance():
    rng = np.random.default_rng(1)
    fix = np.zeros((60, 80), bool)
    fix[rng.integers(0, 60, 40), rng.integers(0, 80, 40)] = True
    gt = maps.saliency_map(pd.DataFrame({"x": np.nonzero(fix)[1], "y": np.nonzero(fix)[0]}), shape=(60, 80), sigma=3)
    assert metrics.auc_judd(gt, fix) > 0.9
    assert metrics.nss(gt, fix) > 1
    assert metrics.cc(gt, gt) == pytest.approx(1.0)
    assert metrics.sim(gt, gt) == pytest.approx(1.0)
    assert metrics.kld(gt, gt) == pytest.approx(0.0, abs=1e-6)
    flat = np.ones_like(gt)
    assert metrics.nss(flat, fix) == pytest.approx(0.0, abs=1e-6)
    assert metrics.auc_shuffled(gt, fix, fix) == pytest.approx(0.5)
    m = metrics.all_metrics(gt, fix, gt, other_fixmap=fix)
    assert set(m) == {"AUC-Judd", "sAUC", "NSS", "CC", "SIM", "KL"}


# ---------------------------------------------------------------- real data (optional)
@pytest.mark.skipif(not os.environ.get("AVATT_DATA"), reason="set AVATT_DATA to a local copy of the dataset")
def test_real_dataset():
    ds = avatt.load(os.environ["AVATT_DATA"], download_if_missing=False)
    assert len(ds) == 283
    assert int(ds.metadata.n_observers.sum()) == 4505
    assert int(ds.metadata.n_fixations.sum()) == 121426
    iid = ds.ids[0]
    sps = ds.scanpaths(iid)
    assert len(sps) == ds.metadata.loc[iid, "n_observers"]
    rebuilt = maps.fixation_map(sps)
    assert (rebuilt == ds.fixation_map(iid)).mean() > 0.999
    assert ds.video_path(iid).exists()
