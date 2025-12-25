# batch_eval.py
import csv
import os
from pathlib import Path
from detect_image import detect_from_path, load_model_cached  # uses your cached loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dir", required=True, help="validation folder (contains subfolders Fake/Real)")
parser.add_argument("--threshold", type=float, default=0.0201)
parser.add_argument("--invert", action="store_true")
parser.add_argument("--out", default="eval_results.csv")
args = parser.parse_args()

model = load_model_cached(args.model)  # warms cache; detect_from_path will reuse

rows = []
for root, _, files in os.walk(args.dir):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, f)
            try:
                label, conf, raw, pf, pr = detect_from_path(
                    path, model_path=args.model, threshold=args.threshold, invert=args.invert
                )
                rows.append({
                    "file": path,
                    "pred_label": label,
                    "confidence": conf,
                    "raw": raw,
                    "prob_fake": pf,
                    "prob_real": pr
                })
            except Exception as e:
                rows.append({
                    "file": path,
                    "pred_label": "ERROR",
                    "confidence": 0,
                    "raw": 0,
                    "prob_fake": 0,
                    "prob_real": 0,
                    "error": str(e)
                })

with open(args.out, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {args.out}")