# find_best_threshold_fix.py
import argparse, os, sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.activations import swish
from sklearn.metrics import roc_curve, f1_score, accuracy_score, confusion_matrix, classification_report

def preprocess(img, size):
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, 0)

def label_from_folder(name):
    # map common folder names -> 0/1
    n = name.lower()
    if n in ("real","reals","real(0)","0","real_images"):
        return 0
    if n in ("fake","fakes","fake(1)","1","fake_images"):
        return 1
    # fallback: try to parse trailing digit
    try:
        d = int(''.join(ch for ch in n if ch.isdigit()))
        return 1 if d==1 else 0
    except:
        return None

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="path to .keras model")
parser.add_argument("root", help="dataset root (contains class subfolders)")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--invert", action="store_true", help="invert model scores (if model gives high for real)")
args = parser.parse_args()

MODEL = args.model
ROOT = args.root
IMG = args.img_size
BATCH = args.batch
INVERT = args.invert

print("Loading model:", MODEL)
model = tf.keras.models.load_model(MODEL, custom_objects={"swish": swish})
print("Model loaded.")

# find class subfolders
classes = [d for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT,d))]
if not classes:
    print("No class subfolders found under", ROOT)
    sys.exit(1)
print("Found subfolders:", classes)

# collect files and numeric labels
files = []
labels = []
for cname in classes:
    lab = label_from_folder(cname)
    if lab is None:
        print(f"WARNING: unknown class folder name '{cname}', skipping")
        continue
    folder = os.path.join(ROOT, cname)
    for root,_,fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                files.append(os.path.join(root,f))
                labels.append(lab)

files = np.array(files)
labels = np.array(labels)
print("Found images:", len(files))
if len(files)==0:
    print("No images found under", ROOT)
    sys.exit(1)

unique_labels = set(labels.tolist())
print("Label set in data:", unique_labels)
if len(unique_labels) < 2:
    print("ERROR: dataset contains only one class in the test folder. ROC/threshold search needs both classes.")
    sys.exit(1)

# predict in batches
scores = []
for i in range(0, len(files), BATCH):
    batch_files = files[i:i+BATCH]
    batch_x = []
    for p in batch_files:
        img = cv2.imread(p)
        if img is None:
            # skip unreadable
            batch_x.append(np.zeros((IMG,IMG,3),dtype=np.float32))
        else:
            batch_x.append(cv2.resize(img,(IMG,IMG)).astype(np.float32)/255.0)
    x = np.stack(batch_x, axis=0)
    preds = model.predict(x, verbose=0).reshape(-1)
    if INVERT:
        preds = 1.0 - preds
    scores.extend(preds.tolist())

scores = np.array(scores)
y_true = labels
y_score = scores

# sanity check shapes
print("y_true.shape:", y_true.shape, "y_score.shape:", y_score.shape)

# compute ROC (will fail if only one class, but we checked)
fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)

# find best threshold by F1
best_thr = None
best_f1 = -1.0
best_report = None
for t in thr:
    y_pred = (y_score >= t).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(t)
        best_report = classification_report(y_true, y_pred, output_dict=True)

print(f"Best F1 {best_f1:.4f} at threshold {best_thr:.4f}")
print("Classification report at best threshold:")
print(classification_report(y_true, (y_score>=best_thr).astype(int)))
# save threshold
with open("best_threshold.txt","w") as f:
    f.write(f"{best_thr:.4f}\n")
print("Saved best_threshold.txt")