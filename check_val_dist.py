# check_val_dist.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

base = "split_dataset_final"
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(base,"val"), target_size=(128,128), batch_size=32,
    class_mode="binary", shuffle=False)
labels = val_gen.classes
uniq, counts = np.unique(labels, return_counts=True)
print("Val class distribution (class_index:value -> count):")
for u,c in zip(uniq, counts):
    print(f"  {u} -> {c}")
print("Total val samples:", len(labels))
