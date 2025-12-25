# dump_preds.py
import argparse, os, numpy as np, cv2, tensorflow as tf
from tensorflow.keras.activations import swish

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--root", required=True, help="dataset root with train/val/test folders OR a single folder")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--out", default="preds.npz")
parser.add_argument("--invert", action="store_true")
args = parser.parse_args()

model = tf.keras.models.load_model(args.model, custom_objects={"swish": swish})
IMG = args.img_size

def iter_images(folder):
    for sub in sorted(os.listdir(folder)):
        subp = os.path.join(folder, sub)
        if not os.path.isdir(subp): continue
        for fn in sorted(os.listdir(subp)):
            if fn.lower().endswith((".jpg",".png",".jpeg")):
                yield sub, os.path.join(subp, fn)

def get_scores(folder):
    labels=[]; scores=[]
    for cls, path in iter_images(folder):
        img = cv2.imread(path)
        if img is None: continue
        x = cv2.resize(img, (IMG,IMG)).astype("float32")/255.0
        p = float(model.predict(x[None], verbose=0).ravel()[0])
        if args.invert:
            p = 1.0 - p
        labels.append(1 if cls.lower().startswith("fake") else 0)
        scores.append(p)
    return np.array(labels), np.array(scores)

out = {}
# if root contains train/val/test directories, process them
for split in ("train","val","test"):
    path = os.path.join(args.root, split)
    if os.path.isdir(path):
        print("Processing", path)
        y, s = get_scores(path)
        out[f"y_{split}"] = y
        out[f"s_{split}"] = s

# also try root as single folder
if not out:
    print("Processing single folder root:", args.root)
    y,s = get_scores(args.root)
    out["y_all"]=y; out["s_all"]=s

np.savez_compressed(args.out, **out)
print("Saved:", args.out)