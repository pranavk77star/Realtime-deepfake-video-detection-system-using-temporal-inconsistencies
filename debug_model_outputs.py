# debug_model_outputs.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "final_best_model_balanced_v2.keras")
TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "balanced", "train")

IMG_SIZE = 128
BATCH_SIZE = 8

print("Loading model:", MODEL_PATH)
m = load_model(MODEL_PATH, compile=False)
print("Model loaded. Last layer name:", m.layers[-1].name)
try:
    print("Last layer output shape (if available):", m.layers[-1].output_shape)
except Exception:
    print("Could not read last layer.output_shape")

# create simple generator but only fetch 1 batch
datagen = ImageDataGenerator(rescale=1.0/255.0)
gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

x_batch, y_batch = next(gen)
print("x_batch.shape:", x_batch.shape)
print("y_batch.shape:", y_batch.shape)
print("y_batch sample:", y_batch[:12])

# forward pass
preds = m.predict(x_batch, verbose=0)
preds = np.asarray(preds)
print("preds.shape:", preds.shape)
print("preds (first 8):", preds.flatten()[:8])
print("preds min/max/mean:", float(np.nanmin(preds)), float(np.nanmax(preds)), float(np.nanmean(preds)))
print("any NaN:", np.isnan(preds).any(), "any inf:", np.isinf(preds).any())

# compute BCE under two assumptions
bce_probs = BinaryCrossentropy(from_logits=False)
bce_logits = BinaryCrossentropy(from_logits=True)

# if preds shape is (batch,1) flatten to (batch,)
p_flat = preds.flatten()

try:
    loss_prob = float(bce_probs(y_batch, p_flat).numpy())
except Exception as e:
    loss_prob = f"error: {e}"

try:
    loss_logits = float(bce_logits(y_batch, p_flat).numpy())
except Exception as e:
    loss_logits = f"error: {e}"

print("BCE (from_logits=False) =>", loss_prob)
print("BCE (from_logits=True)  =>", loss_logits)

# Quick check: if preds outside [0,1] but sigmoided version is inside:
sig = 1.0 / (1.0 + np.exp(-p_flat))
print("sigmoid(min/max):", float(sig.min()), float(sig.max()))