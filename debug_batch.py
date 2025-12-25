# debug_batch.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, sys, numpy as np

base = "split_dataset_final"
IMG = 128
B = 32

def run():
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")

    print("-> Loading generators (this does not train anything)...")
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_dir, target_size=(IMG,IMG), color_mode="rgb", batch_size=B,
        class_mode="binary", shuffle=True, seed=42)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_dir, target_size=(IMG,IMG), color_mode="rgb", batch_size=B,
        class_mode="binary", shuffle=False)

    print("\nClass indices (train):", train_gen.class_indices)
    print("Class indices (val)  :", val_gen.class_indices)
    print("Train samples:", train_gen.samples, "Val samples:", val_gen.samples)

    print("\n-- Inspect one batch from TRAIN generator --")
    x_train, y_train = next(train_gen)
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    print("y_train unique values and counts:", {v:int((y_train==v).sum()) for v in np.unique(y_train)})
    print("x_train batch min/max/mean:", float(x_train.min()), float(x_train.max()), float(x_train.mean()))
    for i in range(min(6, x_train.shape[0])):
        print(f" img[{i}] min/max/mean:", float(x_train[i].min()), float(x_train[i].max()), float(x_train[i].mean()))

    # filenames (generator stores filenames list)
    if hasattr(train_gen, "filenames"):
        print("\nExample train filenames (first 10):")
        for p in train_gen.filenames[:10]:
            print(" ", p)

    print("\n-- Inspect one batch from VAL generator --")
    x_val, y_val = next(val_gen)
    print("x_val shape:", x_val.shape, "y_val shape:", y_val.shape)
    print("y_val unique values and counts:", {v:int((y_val==v).sum()) for v in np.unique(y_val)})
    print("x_val batch min/max/mean:", float(x_val.min()), float(x_val.max()), float(x_val.mean()))

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print("ERROR:", e)
        raise
