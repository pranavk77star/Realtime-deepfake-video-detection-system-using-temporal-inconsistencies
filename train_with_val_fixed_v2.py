#!/usr/bin/env python3
"""
train_with_val_fixed_v2.py
Robust, self-contained training script for EfficientNetB0 binary classifier.
Designed to gracefully handle ImageNet weight channel mismatches and common
dataset issues (missing val class, single-class batches early, etc).
"""
import os
import math
import argparse
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
from sklearn.utils.class_weight import compute_class_weight

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="split_dataset_final")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs_head", type=int, default=3)
parser.add_argument("--epochs_fine", type=int, default=2)
parser.add_argument("--out", type=str, default="tmp_checkpoint/final_best_model.keras")
parser.add_argument("--auto_val", action="store_true", help="If val missing class, auto-move 15%% from train to val")
parser.add_argument("--lr_head", type=float, default=5e-4, help="Learning rate for head training (stage 1)")
parser.add_argument("--lr_fine", type=float, default=1e-5, help="Learning rate for fine-tuning (stage 2)")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- paths & quick checks ----------
DATA_DIR = Path(args.data_dir)
if not DATA_DIR.exists():
    raise SystemExit(f"Data dir not found: {DATA_DIR}")

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
(train_dir / "real").mkdir(parents=False, exist_ok=True)
(train_dir / "fake").mkdir(parents=False, exist_ok=True)
(val_dir / "real").mkdir(parents=False, exist_ok=True)
(val_dir / "fake").mkdir(parents=False, exist_ok=True)

def count_files(p):
    p = Path(p)
    if not p.exists(): return 0
    return sum(1 for _ in p.iterdir() if _.is_file())

print(f"\nData dir: {DATA_DIR}")
print(f"Train real: {count_files(train_dir/'real')} | Train fake: {count_files(train_dir/'fake')}")
print(f"Val   real: {count_files(val_dir/'real')} | Val   fake: {count_files(val_dir/'fake')}\n")

# If val totally missing one class and user asked auto_val, move a small fraction
def ensure_val_has_both():
    r = count_files(val_dir/"real")
    f = count_files(val_dir/"fake")
    if r == 0 or f == 0:
        print("⚠️ Validation split missing a class:", {"real": r, "fake": f})
        if args.auto_val:
            print("-> Auto-splitting ~15% from train into val to create both classes.")
            frac = 0.15
            for cls in ("real","fake"):
                src = train_dir/cls
                dst = val_dir/cls
                dst.mkdir(parents=True, exist_ok=True)
                files = [f for f in src.iterdir() if f.is_file()]
                n_move = max(1, int(len(files) * frac))
                chosen = random.sample(files, n_move)
                for f in chosen:
                    shutil.move(str(f), dst / f.name)
            print("✅ Auto val split created.")
        else:
            print("→ Use --auto_val to auto-create a validation split with both classes.")
    else:
        if args.debug:
            print("Validation appears to have both classes.")

ensure_val_has_both()

# ---------- data generators ----------
IMG_SIZE = args.img_size
BATCH = args.batch

train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=12,
    width_shift_range=0.06,
    height_shift_range=0.06,
    shear_range=0.04,
    zoom_range=0.08,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15),
    fill_mode="reflect"
)
val_datagen = ImageDataGenerator(rescale=1./255.)

train_gen = train_datagen.flow_from_directory(
    str(train_dir),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True,
    seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    str(val_dir),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("\nClass indices (train):", train_gen.class_indices)
print("Class indices (val):  ", val_gen.class_indices)
print("train samples:", train_gen.samples, "val samples:", val_gen.samples, "\n")

# quick debug: show label distribution across first few val batches
if args.debug:
    from collections import Counter
    n_dbg = 6
    print("Debug: checking the labels in the first", n_dbg, "val batches (may be single-class in early batches):")
    counter = Counter()
    for i in range(n_dbg):
        X, y = next(val_gen)
        vals, counts = np.unique(y.astype(int), return_counts=True)
        counter.update(dict(zip(vals.tolist(), counts.tolist())))
        print(f" batch {i}: unique labels={vals.tolist()} counts={counts.tolist()}")
    print("Aggregated counts over these batches:", dict(counter))
    # reset the generator iterator by recreating (since we consumed some)
    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="binary",
        shuffle=False
    )
    print()

# compute class weights (train)
y_train = train_gen.classes
if len(np.unique(y_train)) > 1:
    classes_ = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes_, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes_, cw)}
else:
    class_weights = {0: 1.0, 1: 1.0}

print("Computed class weights:", class_weights, "\n")

# ---------- model building (EfficientNetB0) ----------
input_shape = (IMG_SIZE, IMG_SIZE, 3)
print(f"Building EfficientNetB0 (input={input_shape})")

from tensorflow.keras.applications import EfficientNetB0

weights_loaded = False
base = None

# Attempt A: load with imagenet via input_tensor (best)
try:
    inp = Input(shape=input_shape, name="input_layer")
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
    print("✅ Loaded ImageNet weights via weights='imagenet' + input_tensor.")
    weights_loaded = True
except Exception as e:
    print("⚠️ Could not load ImageNet weights directly:", e)
    # Attempt B: build model weights=None then load cached h5 by_name skipping mismatches
    try:
        inp = Input(shape=input_shape, name="input_layer")
        base = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
        weights_path = get_file('efficientnetb0_notop.h5',
                                'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
                                cache_subdir='.keras/models')
        print("Attempting partial weight load by_name from:", weights_path)
        base.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ Loaded matching ImageNet weights by_name (skipping mismatches).")
        weights_loaded = True
    except Exception as e2:
        print("⚠️ Partial load failed:", e2)
        print(" -> Will build backbone with random initialization (weights=None).")
        inp = Input(shape=input_shape, name="input_layer")
        base = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
        weights_loaded = False

# Freeze base if we have pretrained weights — sensible default
base.trainable = bool(weights_loaded)
print(f"base.trainable set to: {base.trainable} (weights_loaded={weights_loaded})")

# Build head
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base.output)
x = layers.BatchNormalization(name="batch_norm_head")(x)
x = layers.Dropout(0.3, name="dropout_head")(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_512")(x)
x = layers.BatchNormalization(name="batch_norm_512")(x)
x = layers.Dropout(0.25, name="dropout_512")(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
x = layers.Dropout(0.2, name="dropout_128")(x)
out = layers.Dense(1, activation="sigmoid", name="pred")(x)

model = models.Model(inputs=base.input, outputs=out, name="EffNetB0_binary")

# compile (stage 1 lr)
lr_head = args.lr_head
model.compile(optimizer=optimizers.Adam(learning_rate=lr_head),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# model summary (top lines)
print("\nModel built. Summary (top lines):")
model.summary(line_length=120)

# ---------- callbacks ----------
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
ckpt = callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)

steps_train = max(1, math.ceil(train_gen.samples / BATCH))
steps_val = max(1, math.ceil(val_gen.samples / BATCH))
print(f"\nTrain steps: {steps_train} | Val steps: {steps_val} | Batch: {BATCH}")
print(f"Stage 1: epochs_head={args.epochs_head}, lr_head={lr_head}\n")

# ---------- Stage 1: train head (base frozen if pretrained) ----------
print("🧠 Stage 1 training — head (base.trainable = {}) for {} epochs...".format(base.trainable, args.epochs_head))
history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=args.epochs_head,
    class_weight=class_weights,
    callbacks=[ckpt, es, rlr],
    verbose=1
)

# ---------- Stage 2: fine-tune ----------
if args.epochs_fine and args.epochs_fine > 0:
    print("\n🔓 Unfreezing top layers for fine-tuning...")
    # Unfreeze last 30 layers (safe heuristic)
    base.trainable = True
    fine_tune_at = max(0, len(base.layers) - 30)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at
    print(f" Fine-tuning from layer {fine_tune_at} (0-based index) onward — total layers: {len(base.layers)}")

    # lower lr for fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr_fine),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    print("🎯 Fine-tuning for", args.epochs_fine, "epochs...")
    history2 = model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        validation_data=val_gen,
        validation_steps=steps_val,
        epochs=args.epochs_fine,
        class_weight=class_weights,
        callbacks=[ckpt, es, rlr],
        verbose=1
    )

print(f"\n✅ Training complete (or stopped by EarlyStopping). Best model saved to: {args.out}\n")
