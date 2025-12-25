# evaluate_dataset.py
import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import swish
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("root", help="Root dataset folder (contains 'Real' and 'Fake' subfolders)")
parser.add_argument("--model", default="outputs/final_best_model.keras", help="Keras model path")
parser.add_argument("--batch", type=int, default=64, help="Batch size for prediction")
parser.add_argument("--img_size", type=int, default=128, help="Model input size")
parser.add_argument("--invert", action="store_true", help="Invert outputs (use if model gives high=real)")
parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (>= => Fake)")
args = parser.parse_args()

ROOT = args.root
MODEL = args.model
BATCH = args.batch
IMG_SIZE = args.img_size
INVERT = args.invert
TH = args.threshold

print("Model:", MODEL)
print("Root:", ROOT, "Batch size:", BATCH, "Img size:", IMG_SIZE, "Invert:", INVERT, "Threshold:", TH)

print("Loading model...")
model = tf.keras.models.load_model(MODEL, custom_objects={"swish": swish})
print("Model loaded.")

def preprocess_path(p):
    img = cv2.imread(p)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")/255.0
    return img

# gather files
pairs = []  # list of (filepath, label_int)
for label_name, lab in [("Real", 0), ("real", 0), ("Fake", 1), ("fake", 1), ("FF_original", 0), ("FF_allfake",1)]:
    folder = os.path.join(ROOT, label_name)
    if not os.path.isdir(folder):
        continue
    for root_dir, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith((".jpg",".jpeg",".png")):
                continue
            pairs.append((os.path.join(root_dir, fn), lab))

if len(pairs) == 0:
    print("No images found under", ROOT)
    sys.exit(1)

print("Found images:", len(pairs))

# predict in batches
y_true = []
y_score = []

N = len(pairs)
i = 0
batch_imgs = []
batch_lbls = []
while i < N:
    batch_imgs = []
    batch_lbls = []
    for j in range(i, min(i+BATCH, N)):
        p, lab = pairs[j]
        img = preprocess_path(p)
        if img is None:
            continue
        batch_imgs.append(img)
        batch_lbls.append(lab)
    if len(batch_imgs) == 0:
        i += BATCH
        continue
    x = np.stack(batch_imgs, axis=0)
    preds = model.predict(x, verbose=0).ravel()
    # ensure numeric list
    for s, lab in zip(preds, batch_lbls):
        score = float(s)
        pred_prob = 1.0 - score if INVERT else score  # pred_prob = probability for REAL if model outputs high=real
        # We want pred_prob as "probability of Real". For metrics we use label mapping Real=0, Fake=1.
        # Convert to probability of Fake for consistency: p_fake = 1 - p_real
        p_fake = 1.0 - pred_prob
        y_true.append(int(lab))       # 0 or 1
        y_score.append(float(p_fake)) # probability of class=1 (Fake)
    i += BATCH
    if len(y_true) % (BATCH*10) == 0:
        print("Processed", len(y_true), "images...")

y_true = np.array(y_true, dtype=int)
y_score = np.array(y_score, dtype=float)

# predictions
y_pred = (y_score >= TH).astype(int)

print("Done. Total:", len(y_true))
# metrics
try:
    auc = roc_auc_score(y_true, y_score, pos_label=1)
except Exception as e:
    auc = None
    print("ROC calc failed:", e)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred, target_names=["Real(0)","Fake(1)"])

print(f"ROC AUC: {auc}")
print(f"Threshold = {TH:.4f}")
print(f"Accuracy: {acc:.4f}")
print("Confusion matrix (rows=true Real,Fake):")
print(cm)
print("Classification report:")
print(cr)