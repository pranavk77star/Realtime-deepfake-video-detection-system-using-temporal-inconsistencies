# make_val_split.py
"""
Create a validation split by moving `frac` fraction from train -> val.
Usage:
  python make_val_split.py --data_dir split_dataset --frac 0.15
"""
import os, shutil, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="split_dataset")
parser.add_argument("--frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
DATA = args.data_dir
TRAIN = os.path.join(DATA, "train")
VAL = os.path.join(DATA, "val")

if not os.path.isdir(TRAIN):
    raise SystemExit("Train folder not found: " + TRAIN)

os.makedirs(VAL, exist_ok=True)
for cls in ("real","fake"):
    src = os.path.join(TRAIN, cls)
    dst = os.path.join(VAL, cls)
    os.makedirs(dst, exist_ok=True)
    if not os.path.isdir(src):
        print("Warning: missing", src)
        continue
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src,f))]
    n_move = max(1, int(len(files) * args.frac))
    if n_move == 0:
        print(f"Class {cls}: no files to move (total {len(files)})")
        continue
    chosen = random.sample(files, n_move)
    print(f"Moving {n_move} files from {src} -> {dst} (class={cls})")
    for f in chosen:
        shutil.move(os.path.join(src,f), os.path.join(dst,f))

print("Done. Please check split_dataset/val and split_dataset/train counts.")
