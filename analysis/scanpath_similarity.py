"""Scanpath similarity: intra-image (two observers, same painting) vs inter-image baseline.

Usage (from the repository root):  python analysis/scanpath_similarity.py   (after composition_detect.py)
Requires: pip install fdiss eyefeatures   (eyefeatures needs Python 3.10-3.12)
Writes analysis/similarity/ scanpaths.csv, pairs.csv, scores_spatial.csv.
Statistics and figures: similarity_stats.py. Image-encoder metrics: scanpath_embeddings.py.

Design (shared by every metric)
  For each painting I, up to MAX_PAIRS pairs of observers (A, B) who both viewed I       -> intra
  and, for each, a baseline pair (A, C->I): C is a random observer of another painting of
  the same exposure duration, whose fixations are mapped into I by painting coordinates
  (same relative positions inside the painting).                                       -> inter
  The baseline keeps viewing habits (centre bias, scanning order, number of fixations) and
  sits on the same painting, so content-based metrics cannot win just because the
  paintings differ. All scanpaths are cut to their first 5 s.
Spatial metrics: FDISS (1-degree discs) and the eyefeatures distances; eyefeatures takes
coordinates in the unit square (x / 1600, y / 900).
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from eyefeatures.features import dist as ef
from fdiss import FDISS

from analysis import H, OUT, W
from paintfix import load_all

SIM = os.path.join(OUT, "similarity")
WINDOW_MS = 5000
MAX_PAIRS = 45
SCANMATCH_BIN_MS = 50
SEED = 0

# name -> (higher_is_more_similar, label)
METRICS = {
    "FDISS": (True, "FDISS"),
    "MM_shape": (True, "MultiMatch shape"),
    "MM_angle": (True, "MultiMatch direction"),
    "MM_length": (True, "MultiMatch length"),
    "MM_position": (True, "MultiMatch position"),
    "MM_duration": (True, "MultiMatch duration"),
    "DTW": (False, "DTW"),
    "Frechet": (False, "Discrete Fréchet"),
    "Hausdorff": (False, "Hausdorff"),
    "Euclidean": (False, "Euclidean"),
    "Mannan": (False, "Mannan"),
    "EyeAnalysis": (False, "EyeAnalysis"),
    "TDE": (False, "Time-delay embedding"),
    "ScanMatch": (False, "ScanMatch"),
}


def build_design():
    det, fix, ppd = load_all()
    rng = np.random.default_rng(SEED)
    fix = fix[fix.onset_ms < WINDOW_MS]
    box = {n: d["crop_box"] for n, d in det.items()}
    coll = fix.groupby("image").collection.first()
    groups = {k: g for k, g in fix.groupby(["image", "observer"])}
    keys_by_image = {}
    for im, o in groups:
        keys_by_image.setdefault(im, []).append((im, o))

    sp_rows, pair_rows, sid_of = [], [], {}

    def add_scanpath(context, key, mapped):
        sid = len(sid_of)
        sid_of[(context, key, mapped)] = sid
        g = groups[key]
        if mapped:
            x0, y0, x1, y1 = box[context]
            x = np.clip(np.round(x0 + g.u.to_numpy() * (x1 - x0)), 0, W - 1)
            y = np.clip(np.round(y0 + g.v.to_numpy() * (y1 - y0)), 0, H - 1)
        else:
            x, y = g.x.to_numpy(), g.y.to_numpy()
        sp_rows.append(pd.DataFrame({"sid": sid, "image": context, "source_image": key[0], "observer": key[1],
                                     "x": x.astype(int), "y": y.astype(int),
                                     "duration_ms": g.duration_ms.to_numpy(), "onset_ms": g.onset_ms.to_numpy()}))
        return sid

    for im, keys in sorted(keys_by_image.items()):
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        if len(pairs) > MAX_PAIRS:
            pairs = [pairs[i] for i in rng.choice(len(pairs), MAX_PAIRS, replace=False)]
        others = [k for k in groups if k[0] != im and coll[k[0]] == coll[im]]
        for a, b in pairs:
            c = others[rng.integers(len(others))]
            sa = sid_of.get((im, a, False)) if (im, a, False) in sid_of else add_scanpath(im, a, False)
            sb = sid_of.get((im, b, False)) if (im, b, False) in sid_of else add_scanpath(im, b, False)
            sc = sid_of.get((im, c, True)) if (im, c, True) in sid_of else add_scanpath(im, c, True)
            pair_rows += [{"image": im, "collection": coll[im], "kind": "intra", "sid_a": sa, "sid_b": sb},
                          {"image": im, "collection": coll[im], "kind": "inter", "sid_a": sa, "sid_b": sc}]
    os.makedirs(SIM, exist_ok=True)
    sps = pd.concat(sp_rows, ignore_index=True)
    pairs = pd.DataFrame(pair_rows)
    pairs.insert(0, "pair_id", range(len(pairs)))
    sps.to_csv(os.path.join(SIM, "scanpaths.csv"), index=False)
    pairs.to_csv(os.path.join(SIM, "pairs.csv"), index=False)
    with open(os.path.join(SIM, "ppd.txt"), "w") as f:
        f.write(f"{ppd:.4f}\n")
    return sps, pairs, ppd


def compare(job):
    p, q, r_px = job
    pxy, qxy = p[["x", "y"]], q[["x", "y"]]
    out = {"FDISS": FDISS().evaluate(np.c_[p.x * W, p.y * H], np.c_[q.x * W, q.y * H], r_px)["fdiss"]}
    mm = ef.calc_mm_features(p, q)
    out.update(dict(zip(["MM_shape", "MM_angle", "MM_length", "MM_position", "MM_duration"], mm)))
    out["DTW"] = ef.calc_dtw_dist(pxy, qxy)
    out["Frechet"] = ef.calc_dfr_dist(pxy, qxy)
    out["Hausdorff"] = ef.calc_hau_dist(pxy.copy(), qxy.copy())   # shuffles its inputs in place
    out["Euclidean"] = ef.calc_euc_dist(pxy, qxy)
    out["Mannan"] = ef.calc_man_dist(pxy, qxy)
    out["EyeAnalysis"] = ef.calc_eye_dist(pxy, qxy)
    out["TDE"] = ef.calc_tde_dist(pxy, qxy)
    out["ScanMatch"] = ef.calc_scan_match_dist(p, q, t_bin=SCANMATCH_BIN_MS)
    return out


def main():
    sps, pairs, ppd = build_design()
    print(f"{len(pairs)} comparisons ({(pairs.kind == 'intra').sum()} intra-image pairs), "
          f"{sps.sid.nunique()} scanpaths, FDISS radius = {ppd:.2f} px", flush=True)
    paths = {sid: pd.DataFrame({"x": g.x.to_numpy() / W, "y": g.y.to_numpy() / H,
                                "duration": g.duration_ms.to_numpy(float)})
             for sid, g in sps.groupby("sid")}
    jobs = [(paths[a], paths[b], ppd) for a, b in zip(pairs.sid_a, pairs.sid_b)]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(compare, jobs, chunksize=50)
    scores = pd.concat([pairs[["pair_id"]], pd.DataFrame(results)], axis=1)
    scores.to_csv(os.path.join(SIM, "scores_spatial.csv"), index=False)
    print("spatial metrics done", flush=True)


if __name__ == "__main__":
    main()
