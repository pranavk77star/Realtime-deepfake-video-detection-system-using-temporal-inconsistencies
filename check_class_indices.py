# check_class_indices.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

base = "split_dataset_final"
IMG = 128
BATCH = 32

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(base, "train"),
    target_size=(IMG, IMG),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(base, "val"),
    target_size=(IMG, IMG),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("train class_indices:", train_gen.class_indices)
print("val   class_indices:", val_gen.class_indices)
