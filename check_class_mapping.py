# check_class_mapping.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
val_dir = "datasets/balanced/val"
gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(128,128), batch_size=16, class_mode="binary", shuffle=False)
print("class_indices:", gen.class_indices)
print("counts per class:", np.bincount(gen.classes))
print("first 20 labels:", gen.classes[:20])