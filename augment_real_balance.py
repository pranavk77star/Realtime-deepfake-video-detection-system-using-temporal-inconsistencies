#!/usr/bin/env python3
"""
augment_real_balance.py

Augment images in <data_dir>/train/real until the number of real images
is approximately equal to the number of fake images in <data_dir>/train/fake.

Usage:
  python augment_real_balance.py --data_dir split_dataset_final --img_size 128 --seed 42

By default the script will write augmented images back into the train/real folder
with filenames starting with "aug_". Use --out_dir to save to a different folder.
"""
import os
import argparse
import math
import random
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="split_dataset_final",
                    help="root dataset dir (train/val/test subfolders)")
parser.add_argument("--img_size", type=int, default=128, help="target square image size")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_per_image", type=int, default=50,
                    help="maximum augmented images to produce from a single original (safety limit)")
parser.add_argument("--out_dir", type=str, default=None,
                    help="optional: save augmented images to a separate directory (default: overwrite train/real)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

DATA_DIR = args.data_dir
IMG_SIZE = args.img_size
TRAIN_REAL = os.path.join(DATA_DIR, "train", "real")
TRAIN_FAKE = os.path.join(DATA_DIR, "train", "fake")

if args.out_dir:
    OUT_REAL_DIR = args.out_dir
    os.makedirs(OUT_REAL_DIR, exist_ok=True)
else:
    OUT_REAL_DIR = TRAIN_REAL

def list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]

real_files = list_image_files(TRAIN_REAL)
fake_files = list_image_files(TRAIN_FAKE)

n_real = len(real_files)
n_fake = len(fake_files)

print(f"Found real: {n_real} images  |  fake: {n_fake} images")
if n_real == 0:
    raise SystemExit("No images found in train/real — check data_dir or train folder structure.")

if n_real >= n_fake:
    print("Already balanced (real >= fake). No augmentation needed.")
    raise SystemExit(0)

needed = n_fake - n_real
print(f"Need to generate approximately {needed} augmented real images to balance classes.")

# Augmentation settings (tweak if desired)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.06,
    zoom_range=0.14,
    horizontal_flip=True,
    brightness_range=(0.7, 1.35),
    fill_mode="reflect"
)

# Determine unique starting counter so we don't clash with existing 'aug_' files
def _get_start_counter(target_dir):
    existing = list_image_files(target_dir)
    max_idx = -1
    for name in existing:
        if name.startswith("aug_"):
            root = os.path.splitext(name)[0]
            parts = root.split("_")
            if len(parts) >= 3:
                try:
                    idx = int(parts[-1])
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
    return max_idx + 1

counter = _get_start_counter(OUT_REAL_DIR)

# We'll distribute the augmentation across existing real images
per_image = math.ceil(needed / max(1, n_real))
per_image = min(per_image, args.max_per_image)
print(f"Generating up to {per_image} augmentations per original image (max_per_image={args.max_per_image}).")

generated = 0

# Note: we iterate over the snapshot real_files (so newly-created aug files won't be processed)
for fname in real_files:
    if generated >= needed:
        break

    img_path = os.path.join(TRAIN_REAL, fname)
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = img_to_array(im)  # dtype=float32 or uint8 depending on Keras version
            # ensure numeric range is 0..255 (some systems produce floats 0..255)
            if arr.dtype.kind == "f":
                arr = np.clip(arr, 0, 255)
            arr = arr.astype("uint8")
            arr = arr.reshape((1,) + arr.shape)  # shape (1, H, W, C)
    except Exception as e:
        print("Failed to load:", img_path, e)
        continue

    gen_iter = datagen.flow(arr, batch_size=1, shuffle=True)

    # how many to produce from this original
    to_make = min(per_image, needed - generated)
    for i in range(to_make):
        try:
            batch = next(gen_iter)
        except StopIteration:
            break

        # batch is array shape (1,h,w,3) and may be float. Clip & convert to uint8
        img_arr = batch[0]
        img_arr = np.clip(img_arr, 0, 255).astype("uint8")

        out_name = f"aug_{os.path.splitext(fname)[0]}_{counter:06d}.jpg"
        out_path = os.path.join(OUT_REAL_DIR, out_name)

        # if a file with same name somehow exists, bump counter until unique
        while os.path.exists(out_path):
            counter += 1
            out_name = f"aug_{os.path.splitext(fname)[0]}_{counter:06d}.jpg"
            out_path = os.path.join(OUT_REAL_DIR, out_name)

        try:
            Image.fromarray(img_arr).save(out_path, quality=90)
        except Exception as e:
            print("Failed to save:", out_path, e)
            # skip but still advance counters so we don't loop forever
            counter += 1
            continue

        generated += 1
        counter += 1

        if generated % 100 == 0 or generated == needed:
            print(f"Generated {generated}/{needed} augmentations...")

print(f"Done. Generated {generated} augmented real images. Final counts -> real: {n_real + generated}  fake: {n_fake}")
if OUT_REAL_DIR != TRAIN_REAL:
    print("Note: you saved augmented images to a separate folder. Move/copy them into train/real before training.")