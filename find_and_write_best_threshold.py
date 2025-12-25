# find_and_write_best_threshold.py
import os, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, roc_auc_score

MODEL = r"outputs/final_best_model_balanced_v2.keras"
VAL_DIR = r"C:\Users\prana\deepfake_detector\datasets\balanced\val"
IMG_SIZE = 128
BATCH_SIZE = 16
OUT_FILE = "best_threshold.txt"
INVERT = False  # keep False since your eval showed correct mapping without invert

def load_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return tf.keras.models.load_model(path, custom_objects={"swish": tf.nn.swish}, compile=False)

print("Loading model...")
m = load_model(MODEL)
print("Model loaded.")

datagen = ImageDataGenerator(rescale=1.0/255.0)
gen = datagen.flow_from_directory(VAL_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)
n = gen.samples
steps = int(math.ceil(n / BATCH_SIZE))
print("Predicting on validation set...")
preds = m.predict(gen, steps=steps, verbose=1)
# convert preds -> prob_fake
preds = np.asarray(preds)
if preds.ndim == 2 and preds.shape[1] >= 2:
    prob_fake = preds[:,1][:n]
else:
    flat = preds.ravel()[:n]
    prob_fake = (1.0 - flat) if INVERT else flat

y = gen.classes[:n]

# sweep thresholds
best_f1 = -1.0
best_t = 0.5
for t in [i/100 for i in range(1,100)]:
    ypred = (prob_fake >= t).astype(int)
    f1 = f1_score(y, ypred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"Best F1 {best_f1:.4f} at threshold {best_t:.2f}")
# also show AUC
try:
    auc = roc_auc_score(y, prob_fake)
    print("AUC:", auc)
except Exception:
    auc = None

# write to file
with open(OUT_FILE, "w") as fh:
    fh.write(str(best_t))
print(f"Wrote threshold to {OUT_FILE}")