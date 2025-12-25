# check_weights.py
import h5py
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Input
import os

weights_path = r"C:\Users\prana\.keras\.keras\models\efficientnetb0_notop.h5"  # <- update if different
print("weights_path exists:", os.path.exists(weights_path))

# build the model (weights=None so we don't re-download)
inp = Input(shape=(128, 128, 3))
model = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)

with h5py.File(weights_path, "r") as f:
    if "model_weights" in f:
        groups = list(f["model_weights"].keys())
    else:
        # older layout: top-level keys are layer names
        groups = list(f.keys())

model_layer_names = [layer.name for layer in model.layers]
matched = [g for g in groups if g in model_layer_names]
print(f"Total groups in weights file: {len(groups)}")
print(f"Total layers in model: {len(model_layer_names)}")
print(f"Matched group names (count): {len(matched)}")
print("First 30 matched names:")
for n in matched[:30]:
    print(" ", n)
