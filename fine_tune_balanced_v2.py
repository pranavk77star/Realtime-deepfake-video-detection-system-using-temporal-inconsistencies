# fine_tune_balanced_v2.py
"""
Robust fine-tuning script for the deepfake detector.
- Ensures exactly two training classes (Fake, Real)
- Automatic loss selection (from_logits True/False)
- Computes and caps class weights to avoid instability
- Augmentation, checkpointing, early stopping, LR reduction
- Safe KeyboardInterrupt handling
"""

import os
import sys
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy

# ---------- PATHS & CONFIG ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "balanced", "train")
VAL_DIR = os.path.join(BASE_DIR, "datasets", "balanced", "val")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "final_best_model_balanced_v2.keras")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "fine_tuned_v3.keras")
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-5
CLASS_WEIGHT_CAP = 4.0  # prevents huge unstable weights

print("Config:")
print(" BASE_DIR:", BASE_DIR)
print(" TRAIN_DIR:", TRAIN_DIR)
print(" VAL_DIR:", VAL_DIR)
print(" MODEL_PATH:", MODEL_PATH)
print(" OUTPUT_PATH:", OUTPUT_PATH)
print("IMG_SIZE, BATCH_SIZE, EPOCHS, LR:", IMG_SIZE, BATCH_SIZE, EPOCHS, LR)
print("GPUs available:", tf.config.list_physical_devices("GPU"))

# ---------- sanity checks ----------
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Train dir not found: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Val dir not found: {VAL_DIR}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Base model not found: {MODEL_PATH}")

# check train subfolders
train_subdirs = [d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()]
print("Train subfolders detected:", train_subdirs)
if len(train_subdirs) != 2:
    raise RuntimeError(
        f"Expected exactly 2 subfolders in train dir (Fake, Real). Found {len(train_subdirs)}: {train_subdirs}.\n"
        "Please merge any extras (e.g. Real_extra) into the two class folders before running."
    )

# ---------- load base model ----------
print("\nLoading base model (compile=False)...")
base_model = load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# inspect last layer to decide loss behaviour
has_sigmoid = False
try:
    last_layer = base_model.layers[-1]
    print("Last layer:", last_layer.name)
    try:
        print("  last layer output shape:", last_layer.output_shape)
    except Exception:
        print("  (output_shape not available)")
    act = getattr(last_layer, "activation", None)
    if act is not None and getattr(act, "__name__", "") == "sigmoid":
        has_sigmoid = True
    print("  activation is sigmoid?:", has_sigmoid)
except Exception as e:
    print("Warning inspecting last layer:", e)
    has_sigmoid = False

# choose loss
loss_fn = BinaryCrossentropy(from_logits=not has_sigmoid)
print("Using loss:", loss_fn)

# make entire model trainable for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

base_model.compile(optimizer=Adam(learning_rate=LR), loss=loss_fn, metrics=["accuracy"])

# ---------- data generators ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

print("\nCreating generators...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------- compute class weights (balanced, capped) ----------
try:
    labels = train_gen.classes  # numpy array of 0/1 labels
    counts = Counter(labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise RuntimeError(f"Expected 2 classes; found: {classes}")
    total = float(len(labels))
    class_weights = {}
    for c in classes:
        w = float(total / (len(classes) * counts[int(c)]))
        if w > CLASS_WEIGHT_CAP:
            print(f"Class weight {w:.2f} for class {c} is large — capping to {CLASS_WEIGHT_CAP}.")
            w = float(CLASS_WEIGHT_CAP)
        class_weights[int(c)] = w
    print("Train class counts:", counts)
    print("Using class_weights:", class_weights)
except Exception as e:
    print("Warning computing class weights:", e)
    class_weights = None

# ---------- callbacks ----------
checkpoint_cb = ModelCheckpoint(
    OUTPUT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
)
earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# ---------- training ----------
print("\nStarting fine-tuning...")
try:
    history = base_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb],
        class_weight=class_weights if class_weights is not None else None,
        verbose=1
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user (KeyboardInterrupt).")
    if os.path.exists(OUTPUT_PATH):
        print("A best-checkpoint exists at:", OUTPUT_PATH)
    else:
        print("No checkpoint found. Re-run to continue training.")
    sys.exit(0)
except Exception as e:
    print("Training failed with exception:", e)
    raise

# ---------- save & final eval ----------
print("\nSaving final model to:", OUTPUT_PATH)
base_model.save(OUTPUT_PATH)

print("\nEvaluating on validation set...")
res = base_model.evaluate(val_gen, verbose=1)
print("Validation results (loss, accuracy):", res)

print("\nDone.")