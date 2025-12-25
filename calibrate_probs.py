# calibrate_probs.py
import os, math, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

MODEL = "outputs/final_best_model_balanced_v2.keras"
VAL_DIR = r"C:\Users\prana\deepfake_detector\datasets\balanced\val"
IMG_SIZE = 128
BATCH_SIZE = 16
OUT_FILE = "prob_calibrator.pkl"
METHOD = "istonic"  # "platt" or "isotonic"

print("Loading model...")
model = tf.keras.models.load_model(MODEL, compile=False)
print("Model loaded.")

datagen = ImageDataGenerator(rescale=1.0/255.0)
gen = datagen.flow_from_directory(VAL_DIR, target_size=(IMG_SIZE,IMG_SIZE),
                                  batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)
n = gen.samples
steps = math.ceil(n / BATCH_SIZE)
preds = model.predict(gen, steps=steps, verbose=1)
preds = np.asarray(preds)

# convert preds -> prob_fake
if preds.ndim == 2 and preds.shape[1] >= 2:
    prob_fake = preds[:,1][:n]
else:
    prob_fake = preds.ravel()[:n]

y = gen.classes[:n]

# split to fit calibrator (keep a small heldout part)
X_train, X_val, y_train, y_val = train_test_split(prob_fake.reshape(-1,1), y, test_size=0.2, random_state=42)
if METHOD == "platt":
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    calibrated = clf.predict_proba(X_val)[:,1]
else:
    clf = IsotonicRegression(out_of_bounds='clip')
    clf.fit(X_train.ravel(), y_train)
    calibrated = clf.predict(X_val.ravel())

print("Brier before:", brier_score_loss(y_val, X_val.ravel()))
print("Brier after:", brier_score_loss(y_val, calibrated))

with open(OUT_FILE, "wb") as f:
    pickle.dump(clf, f)
print("Saved calibrator to", OUT_FILE)