"""Run the full analysis pipeline in order (results go to results/, or AVATT_RESULTS).

Usage (from the repository root):  python scripts/run_all.py [--from STEP] [--only STEP ...]
Requires the dataset (python -c "import avatt; avatt.download('data/AVAtt')") and the third-party models
(python scripts/setup_third_party.py). A CUDA GPU with 4 GB is enough; total run time is a few hours on a laptop.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ("descriptive", ["analysis/analysis.py"]),
    ("composition_detect", ["analysis/composition_detect.py"]),
    ("composition_fit", ["analysis/composition_fit.py"]),
    ("temporal", ["analysis/temporal.py"]),
    ("saliency_models", [["analysis/saliency_models.py", m] for m in
                         ["spectral_residual", "salient_objects", "unisal", "transalnet_res", "deepgaze_iie",
                          "ittikoch", "gbvs"]]),
    ("benchmark", ["analysis/benchmark.py"]),
    ("features", ["analysis/features_baseline.py"]),
    ("faces_objects", ["analysis/faces_objects.py"]),
    ("composition_fixations", ["analysis/composition_fixations.py"]),
    ("painting_factors", ["analysis/painting_factors.py"]),
    ("scanpath_similarity", ["analysis/scanpath_similarity.py"]),
    ("scanpath_embeddings", ["analysis/scanpath_embeddings.py"]),
    ("similarity_stats", ["analysis/similarity_stats.py"]),
    ("style", ["analysis/style_analysis.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
    ap.add_argument("--only", nargs="+", choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    names = [s for s, _ in STEPS]
    todo = names[names.index(a.start):] if a.start else names
    if a.only:
        todo = [s for s in names if s in a.only]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "analysis"), os.environ.get("PYTHONPATH", "")]))
    for name, cmds in STEPS:
        if name not in todo:
            continue
        for cmd in cmds:
            cmd = cmd if isinstance(cmd, list) else [cmd]
            t = time.time()
            print(f"== {name}: {' '.join(cmd)}", flush=True)
            subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)
            print(f"   done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
