"""Assemble the AVAtt Hugging Face release from the curated dataset folder.

Usage:
    python scripts/make_release.py --src <curated AVAtt dir> --boxes <detections.json> --out <release dir>

The curated folder (output of scripts/build_from_tobii.py + painting metadata) contains images/, scanpaths/,
fixation_maps/, saliency_heatmaps/, temporal_saliency_maps/ and metadata.csv. The release folder gets:

    images/<id>.jpg                     1600x900 stimulus (painting on grey)
    scanpaths/<id>.jsonl                one line per observer
    fixation_maps/<id>.png              binary fixation map
    saliency_maps/<id>.png              Gaussian-smoothed map, sigma = 1 degree
    temporal_saliency_maps/<id>.mp4     accumulated attention, 60 fps, 5 s
    metadata.csv                        one row per painting
    data/train-*.parquet                everything above except videos, for datasets.load_dataset / the viewer
"""
import argparse
import csv
import io
import json
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image

FIXED_NAMES = {  # original file names corrected during curation (kept in `original_filename` after correction)
    "madame-stora-in-algerian-dress-1870": "wassily-kandinsky-abstraction",   # file showed a Kandinsky, not the Renoir
    "madame-stora-in-algerian-dress-1870_true": "madame-stora-in-algerian-dress-1870",
}
DUPLICATE_OF = {"Las_Meninas,_by_Diego_Velazquez,_from_Prado_in_Google_Earth_big":
                "Las_Meninas,_by_Diego_Velazquez,_from_Prado_in_Google_Earth"}

# death years for copyright status (works enter the public domain 70 years after the artist's death in the EU)
DEATH = {
    "Alfred Sisley": 1899, "Pierre-Auguste Renoir": 1919, "Paul Gauguin": 1903, "Pablo Picasso": 1973,
    "Pieter Bruegel the Elder": 1569, "Salvador Dali": 1989, "Paul Klee": 1940, "Melchior d'Hondecoeter": 1695,
    "Joachim Beuckelaer": 1575, "Canaletto": 1768, "Adriaen van Utrecht": 1652, "Jan van Goyen": 1656,
    "Aelbert Cuyp": 1691, "Aert van der Neer": 1677, "Philips Koninck": 1688, "Salomon van Ruysdael": 1670,
    "Roelant Savery": 1639, "Rosa Bonheur": 1899, "Philips Wouwerman": 1668, "Vincent van Gogh": 1890,
    "Rembrandt van Rijn": 1669, "Edvard Munch": 1944, "Albert Gleizes": 1953, "Claude Lorrain": 1682,
    "Frans Hals": 1666, "Hubert Roestenburg": None, "Ernst Ludwig Kirchner": 1938, "Georges de La Tour": 1652,
    "Diego Velazquez": 1660, "Franz Marc": 1916, "Max Beckmann": 1950, "Nicolas Mignard": 1668,
    "Orazio Gentileschi": 1639, "Salvator Rosa": 1673, "Willem den Broeder": None, "James McNeill Whistler": 1903,
    "Asher Brown Durand": 1886, "Lyubov Popova": 1924, "Benozzo Gozzoli": 1497, "Antonio del Pollaiolo": 1498,
    "Roger de La Fresnaye": 1925, "Hilma af Klint": 1944, "Katsushika Hokusai": 1849, "Ito Jakuchu": 1800,
    "Camille Corot": 1875, "Theo van Doesburg": 1931, "Albert Bierstadt": 1902, "Bada Shanren": 1705,
    "Gentile Bellini": 1507, "Henri Regnault": 1871, "George Stubbs": 1806, "Georges Seurat": 1891,
    "Edouard Manet": 1883, "Adolph Menzel": 1905, "Leonardo da Vinci (attributed)": 1519, "Leonardo da Vinci": 1519,
    "Thomas Dewing": 1938, "Camille Pissarro": 1903, "Wassily Kandinsky": 1944, "Antonello da Messina": 1479,
    "Andrea Mantegna": 1506, "Martin Johnson Heade": 1904, "Paul Signac": 1935, "Victor Hugo": 1885,
    "Paolo Uccello": 1475, "Joseph Wright of Derby": 1797, "Umberto Boccioni": 1916, "Thomas Jones": 1803,
    "Frederic Remington": 1909, "Giovanni Bellini and Titian": 1576, "Theodore Gericault": 1824, "David Allan": 1796,
    "George Inness": 1894, "Kazimir Malevich": 1935, "Giorgio de Chirico": 1978, "Giorgione": 1510, "Raphael": 1520,
    "Juan Gris": 1927, "Alexej von Jawlensky": 1941, "Sophie Taeuber-Arp": 1943,
}
MODERN_UNKNOWN = {"Contemporary", "Modern (unattributed)", "Fauvism"}
PD_CUTOFF = 1955          # died in or before this year -> public domain in 2026 (life + 70)


def copyright_status(row):
    artist = row["artist"] if isinstance(row["artist"], str) else ""
    if artist in DEATH:
        d = DEATH[artist]
        if d is None:
            return "in copyright"
        return "public domain" if d <= PD_CUTOFF else "in copyright"
    if row["movement"] in MODERN_UNKNOWN or row["movement_group"] in {"Abstract art", "Expressionism", "Cubism", "Surrealism"}:
        return "unknown"
    return "public domain"


def png_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--boxes", required=True, help="detections.json with painting crop boxes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=4)
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.src, "metadata.csv"))
    det = json.load(open(a.boxes))
    os.makedirs(a.out, exist_ok=True)
    for d in ["images", "scanpaths", "fixation_maps", "saliency_maps", "temporal_saliency_maps", "data"]:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    # stable ids avatt_0001 ... in order of the (corrected) original file names
    order = sorted(meta["image"], key=lambda o: FIXED_NAMES.get(o, o).lower())
    ID = {o: f"avatt_{i:04d}" for i, o in enumerate(order, 1)}
    rows, records = [], []
    for _, r in meta.iterrows():
        old = r["image"]
        new = ID[old]
        x0, y0, x1, y1 = det[old]["crop_box"]
        copies = [("images", ".jpg", "images", ".jpg"), ("scanpaths", ".jsonl", "scanpaths", ".jsonl"),
                  ("fixation_maps", ".png", "fixation_maps", ".png"), ("saliency_heatmaps", ".png", "saliency_maps", ".png"),
                  ("temporal_saliency_maps", ".mp4", "temporal_saliency_maps", ".mp4")]
        for sd, se, dd, de in copies:
            shutil.copyfile(os.path.join(a.src, sd, old + se), os.path.join(a.out, dd, new + de))
        with open(os.path.join(a.out, "scanpaths", new + ".jsonl")) as f:
            sps = [json.loads(l) for l in f]
        spans = [sp["fixations"][-1]["onset_ms"] + sp["fixations"][-1]["duration_ms"] for sp in sps]
        exposure = 15.0 if np.median(spans) > 8000 else 5.5
        row = {
            "image_id": new, "title": r["title"] if isinstance(r["title"], str) else "",
            "artist": r["artist"] if isinstance(r["artist"], str) else "",
            "year": r["year"] if isinstance(r["year"], str) else ("" if pd.isna(r["year"]) else str(r["year"])),
            "movement": r["movement"], "movement_group": r["movement_group"], "genre": r["genre"],
            "medium": r["medium"] if isinstance(r["medium"], str) else "",
            "attribution_confidence": r["confidence"], "copyright_status": copyright_status(r),
            "exposure_s": exposure, "n_observers": len(sps), "n_fixations": sum(len(sp["fixations"]) for sp in sps),
            "painting_x0": x0, "painting_y0": y0, "painting_x1": x1, "painting_y1": y1,
            "duplicate_of": ID.get(DUPLICATE_OF.get(old, ""), ""),
            "note": r["note"] if isinstance(r["note"], str) and "MISLABELLED" not in r["note"] else "",
            "original_filename": FIXED_NAMES.get(old, old),
        }
        rows.append(row)
        records.append({**row,
                        "image": {"bytes": png_bytes(os.path.join(a.out, "images", new + ".jpg")), "path": new + ".jpg"},
                        "fixation_map": {"bytes": png_bytes(os.path.join(a.out, "fixation_maps", new + ".png")), "path": new + ".png"},
                        "saliency_map": {"bytes": png_bytes(os.path.join(a.out, "saliency_maps", new + ".png")), "path": new + ".png"},
                        "scanpaths": [{"observer": sp["observer"],
                                       "x": [p["x"] for p in sp["fixations"]], "y": [p["y"] for p in sp["fixations"]],
                                       "onset_ms": [p["onset_ms"] for p in sp["fixations"]],
                                       "duration_ms": [p["duration_ms"] for p in sp["fixations"]],
                                       "distance_mm": [p["distance_mm"] for p in sp["fixations"]]} for sp in sps]})

    md = pd.DataFrame(rows).sort_values("image_id")
    md.to_csv(os.path.join(a.out, "metadata.csv"), index=False, quoting=csv.QUOTE_MINIMAL)

    from datasets import Dataset, Features, Image as HFImage, List, Value
    feats = Features({
        "image_id": Value("string"), "image": HFImage(), "fixation_map": HFImage(), "saliency_map": HFImage(),
        "scanpaths": List({"observer": Value("int32"), "x": List(Value("int32")), "y": List(Value("int32")),
                           "onset_ms": List(Value("int32")), "duration_ms": List(Value("int32")),
                           "distance_mm": List(Value("float32"))}),
        "title": Value("string"), "artist": Value("string"), "year": Value("string"), "movement": Value("string"),
        "movement_group": Value("string"), "genre": Value("string"), "medium": Value("string"),
        "attribution_confidence": Value("string"), "copyright_status": Value("string"), "exposure_s": Value("float32"),
        "n_observers": Value("int32"), "n_fixations": Value("int32"), "painting_x0": Value("int32"),
        "painting_y0": Value("int32"), "painting_x1": Value("int32"), "painting_y1": Value("int32"),
        "duplicate_of": Value("string"), "note": Value("string"), "original_filename": Value("string"),
    })
    records.sort(key=lambda r: r["image_id"])
    ds = Dataset.from_list(records, features=feats)
    for i in range(a.shards):
        ds.shard(a.shards, i, contiguous=True).to_parquet(
            os.path.join(a.out, "data", f"train-{i:05d}-of-{a.shards:05d}.parquet"))
    print(f"{len(md)} paintings, {int(md.n_observers.sum())} scanpaths, {int(md.n_fixations.sum())} fixations")
    print(md.copyright_status.value_counts().to_dict())


if __name__ == "__main__":
    main()
