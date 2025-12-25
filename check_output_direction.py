# check_output_direction.py
import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import swish
import cv2

MODEL = "outputs/final_best_model.keras"
VAL_DIR = "split_dataset_final/val"   # adjust if needed
IMG_SIZE = 128
N_SAMPLES = 100

print("Loading model...", MODEL)
model = tf.keras.models.load_model(MODEL, custom_objects={"swish": swish})
print("Model loaded.")

def preprocess(path):
    im = cv2.imread(path)
    if im is None: 
        return None
    h,w = im.shape[:2]
    # center crop if tall/wide irregularities
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    im = im.astype("float32")/255.0
    return np.expand_dims(im,0)

def sample_preds(classname):
    folder = os.path.join(VAL_DIR, classname)
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        raise SystemExit("No files in " + folder)
    files = random.sample(files, min(N_SAMPLES, len(files)))
    preds=[]
    for p in files:
        x = preprocess(p)
        if x is None: continue
        preds.append(float(model.predict(x, verbose=0).ravel()[0]))
    return preds

print("Sampling predictions... (this may take a while)")
p_fake = sample_preds("fake")
p_real = sample_preds("real")

print(f"fake:  mean={np.mean(p_fake):.4f}  median={np.median(p_fake):.4f}  n={len(p_fake)}")
print(f"real:  mean={np.mean(p_real):.4f}  median={np.median(p_real):.4f}  n={len(p_real)}")

# heuristic decision
if np.mean(p_fake) > np.mean(p_real):
    print("\nConclusion: model output HIGHER => more likely FAKE (no inversion needed).")
else:
    print("\nConclusion: model output HIGHER => more likely REAL (you should INVERT predictions).")

print("\nSuggested THRESHOLD (mean of means):", (np.mean(p_fake)+np.mean(p_real))/2.0)