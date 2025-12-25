# debug_batches.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "split_dataset_final"
IMG_SIZE = 128
BATCH = 32

train_gen = ImageDataGenerator(rescale=1./255.).flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255.).flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("Class indices (train):", train_gen.class_indices)
print("Class indices (val):  ", val_gen.class_indices)
print("train samples:", train_gen.samples, "val samples:", val_gen.samples)

# show labels/fn for 3 batches from train and 3 batches from val
def inspect(gen, name, n_batches=3):
    print("\nInspecting", name)
    unique_counts = {}
    for i in range(n_batches):
        X, y = next(gen)
        y = y.astype(int)
        vals, cnts = np.unique(y, return_counts=True)
        print(f" batch {i}: label unique={vals.tolist()}, counts={cnts.tolist()}")
        # print first 6 filenames if available
        if hasattr(gen, "filenames"):
            start = (gen.batch_index - 1) * gen.batch_size
            # handle wrap
            idxs = list(range(start, min(start + gen.batch_size, len(gen.filenames))))
            sample_files = [gen.filepaths[j] if hasattr(gen, "filepaths") else gen.filenames[j] for j in idxs][:6]
            print("  sample files:", sample_files)
    print()

inspect(train_gen, "TRAIN")
inspect(val_gen, "VAL")
