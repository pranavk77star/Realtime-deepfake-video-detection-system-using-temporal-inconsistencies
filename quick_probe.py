# snippet to sample N random val files and predict
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, pandas as pd
from tensorflow.keras.models import load_model

IMG = 128
base = "split_dataset_final"
val_gen_full = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(base,"val"), target_size=(IMG,IMG), color_mode="rgb",
    batch_size=32, class_mode="binary", shuffle=False)

# number to probe (cap)
N = min(200, val_gen_full.samples)
rng = np.random.default_rng(42)
idx = rng.choice(val_gen_full.samples, size=N, replace=False)

# build a generator that yields exactly the selected images in that order:
def make_subset_generator(gen, indices, batch_size=32):
    # load images from filenames via gen.directory + gen.filenames
    base_dir = gen.directory
    fnames = [gen.filenames[i] for i in indices]
    X = []
    y = []
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    for f in fnames:
        img = load_img(os.path.join(base_dir, f), target_size=(IMG,IMG))
        arr = img_to_array(img)/255.0
        X.append(arr)
        label = 1 if f.split(os.sep)[0]=="real" else 0
        y.append(label)
    X = np.stack(X, axis=0)
    y = np.array(y)
    return X, y, fnames

X, y, fnames = make_subset_generator(val_gen_full, idx, batch_size=32)
model = load_model("tmp_checkpoint/final_best_model.keras", compile=False)
probs = model.predict(X, batch_size=32)
df = pd.DataFrame({"filename": fnames, "label": y, "prob_fake": probs.ravel()})
print(df["prob_fake"].describe())
print("predicted fake fraction:", (df["prob_fake"] >= 0.5).mean())
df.to_csv("reports/quick_probe_preds.csv", index=False)
