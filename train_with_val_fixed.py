#!/usr/bin/env python3
"""
train_with_val_fixed.py

Robust, corrected training script for the deepfake detector project.
Features / fixes:
 - robust ImageNet pretrained loading with partial-by-name fallback
 - sensible freezing logic: freeze backbone when pretrained weights were loaded
 - optional CLI overrides (force_freeze / force_unfreeze)
 - reproducible-ish seeds
 - clear diagnostic prints for counts, class indices, generator samples
 - safe callbacks and training stages (head then fine-tune)
 - helpful --debug flag to print extra info

Usage example:
 python train_with_val_fixed.py --data_dir split_dataset_final --img_size 128 --batch 32 \
    --epochs_head 6 --epochs_fine 6 --out tmp_checkpoint/final_best_model.keras --auto_val

"""

import os
import math
import argparse
import random
import shutil
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
from sklearn.utils.class_weight import compute_class_weight

# reproducible-ish
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Force consistent format
tf.keras.backend.set_image_data_format("channels_last")

parser = argparse.ArgumentParser(description="Train EfficientNetB0-based deepfake detector")
parser.add_argument("--data_dir", type=str, default="split_dataset_final")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs_head", type=int, default=6)
parser.add_argument("--epochs_fine", type=int, default=6)
parser.add_argument("--out", type=str, default="tmp_checkpoint/final_best_model.keras")
parser.add_argument("--auto_val", action="store_true", help="Automatically move ~15% of train -> val if val too small")
parser.add_argument("--force_freeze", action="store_true", help="Force backbone frozen during stage1")
parser.add_argument("--force_unfreeze", action="store_true", help="Force backbone trainable during stage1")
parser.add_argument("--debug", action="store_true", help="Print debug info")
args = parser.parse_args()

DATA_DIR = args.data_dir
IMG_SIZE = args.img_size
BATCH = args.batch
OUT_PATH = args.out
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

# helper
def count_dir(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

train_real = os.path.join(DATA_DIR, "train", "real")
train_fake = os.path.join(DATA_DIR, "train", "fake")
val_real = os.path.join(DATA_DIR, "val", "real")
val_fake = os.path.join(DATA_DIR, "val", "fake")

print(f"Train real: {count_dir(train_real)} | Train fake: {count_dir(train_fake)}")
print(f"Val real: {count_dir(val_real)} | Val fake: {count_dir(val_fake)}")

# --- optional auto val creation ---
min_val_required = 30
val_real_count, val_fake_count = count_dir(val_real), count_dir(val_fake)

if (val_real_count < min_val_required or val_fake_count < min_val_required):
    if args.auto_val:
        frac = 0.15
        print("Auto-creating validation split (~15% from train)...")
        for cls in ("real", "fake"):
            src = os.path.join(DATA_DIR, "train", cls)
            dst = os.path.join(DATA_DIR, "val", cls)
            os.makedirs(dst, exist_ok=True)
            files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
            if not files:
                print(f"  warning: no files found in {src}")
                continue
            n_move = max(1, int(len(files) * frac))
            chosen = random.sample(files, n_move)
            for f in chosen:
                shutil.move(os.path.join(src, f), os.path.join(dst, f))
        val_real_count, val_fake_count = count_dir(val_real), count_dir(val_fake)
        print(f"✅ Auto val split created. Now val real={val_real_count}, val fake={val_fake_count}")
    else:
        print(f"⚠️ Warning: Validation set small ({val_real_count}, {val_fake_count}). Continue anyway.")

# --- data generators ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=18,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.06,
    zoom_range=0.12,
    horizontal_flip=True,
    brightness_range=(0.7, 1.35),
    fill_mode="reflect"
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False,
    seed=SEED
)

print("\nClass indices (train):", train_gen.class_indices)
print("Class indices (val):  ", val_gen.class_indices)
print("train samples:", train_gen.samples, "val samples:", val_gen.samples)

# --- compute class weights ---
y_train = train_gen.classes
if len(np.unique(y_train)) > 1:
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(cls): float(w) for cls, w in zip(classes, cw)}
else:
    class_weights = {0: 1.0, 1: 1.0}

print("\nComputed class weights:", class_weights, "\n")

# --- model setup ---
input_shape = (IMG_SIZE, IMG_SIZE, 3)
from tensorflow.keras.applications import EfficientNetB0

print(f"Building EfficientNetB0 (input={input_shape})")
weights_loaded = False
base = None

# attempt to load pretrained weights robustly
try:
    inp = Input(shape=input_shape)
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
    print("✅ Loaded ImageNet weights (direct).")
    weights_loaded = True
except Exception as e:
    print("⚠️ Could not load ImageNet weights directly:", e)
    # try partial by_name load from keras-applications file
    try:
        inp = Input(shape=input_shape)
        base = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
        weights_path = get_file('efficientnetb0_notop.h5',
                                'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
                                cache_subdir='.keras/models')
        print("Attempting partial weight load (by_name=True, skip_mismatch=True) from:", weights_path)
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ Loaded matching ImageNet weights by_name (skipping mismatches).")
        weights_loaded = True
    except Exception as e2:
        print("⚠️ Partial weight load failed:", e2)
        print("   -> Will initialize backbone randomly (weights=None).")
        inp = Input(shape=input_shape)
        base = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
        weights_loaded = False

# decide trainable for stage-1
if args.force_freeze:
    base.trainable = False
elif args.force_unfreeze:
    base.trainable = True
else:
    # Freeze backbone if we have pretrained (direct or partial) weights
    base.trainable = False if weights_loaded else True

print(f"base.trainable set to: {base.trainable} (weights_loaded={weights_loaded})")

# --- head ---
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs=base.input, outputs=out)

# --- compile ---
lr_head = 1e-4
model.compile(optimizer=optimizers.Adam(learning_rate=lr_head), loss="binary_crossentropy", metrics=["accuracy"]) 
model.summary()

# --- callbacks ---
ckpt = callbacks.ModelCheckpoint(OUT_PATH, monitor="val_loss", save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)

steps_train = max(1, math.ceil(train_gen.samples / BATCH))
steps_val = max(1, math.ceil(val_gen.samples / BATCH))

# --- stage 1: train head (or whole network if backbone random) ---
print("\n🧠 Stage 1 training — head (base.trainable = {}) for {} epochs...".format(base.trainable, args.epochs_head))
model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=args.epochs_head,
    class_weight=class_weights,
    callbacks=[ckpt, es, rlr],
    verbose=1
)

# --- stage 2: fine-tune ---
print("\n🔓 Unfreezing top layers for fine-tuning...")
base.trainable = True
# unfreeze last N layers
fine_tune_at = max(0, len(base.layers) - 30)
for i, layer in enumerate(base.layers):
    layer.trainable = i >= fine_tune_at

# lower lr for fine-tune
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"]) 

print("🎯 Fine-tuning for", args.epochs_fine, "epochs...")
model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=args.epochs_fine,
    class_weight=class_weights,
    callbacks=[ckpt, es, rlr],
    verbose=1
)

print(f"\n✅ Training complete! Best model saved to: {OUT_PATH}")

# small helpful debug prints
if args.debug:
    print("\n--- Debug: sample inputs from validation generator ---")
    x_val, y_val = next(iter(val_gen))
    print("x_val shape:", getattr(x_val, 'shape', None), "y_val unique:", np.unique(y_val, return_counts=True))

# exit cleanly
sys.exit(0)
