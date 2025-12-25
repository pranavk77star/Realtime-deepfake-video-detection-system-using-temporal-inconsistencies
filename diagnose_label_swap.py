# diagnose_label_swap.py
import numpy as np
import os, math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

MODEL = r"outputs/final_best_model_balanced_v2.keras"
VAL_DIR = r"C:\Users\prana\deepfake_detector\datasets\balanced\val"
IMG_SIZE = 128
BATCH_SIZE = 16
THRESH = 0.5
INVERT = False   # set True if your model's single-output is prob_real

def load_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        return tf.keras.models.load_model(path, custom_objects={"swish": tf.nn.swish}, compile=False)

model = load_model(MODEL)
datagen = ImageDataGenerator(rescale=1.0/255.0)
gen = datagen.flow_from_directory(VAL_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)
print("class_indices:", gen.class_indices)
n = gen.samples
steps = int(math.ceil(n / BATCH_SIZE))
preds = model.predict(gen, steps=steps, verbose=1)
# convert preds -> prob_fake
preds = np.asarray(preds)
if preds.ndim == 2 and preds.shape[1] >= 2:
    prob_fake = preds[:,1]
else:
    flat = preds.ravel()
    prob_fake = (1.0 - flat) if INVERT else flat

prob_fake = prob_fake[:n]
y = gen.classes

def eval_with_labels(y_true, prob_fake, threshold=THRESH):
    y_pred = (prob_fake >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = None
    try:
        auc = roc_auc_score(y_true, prob_fake)
    except:
        pass
    return acc, prec, rec, f1, cm, auc

print("\n--- Using default gen.classes mapping ---")
acc,prec,rec,f1,cm,auc = eval_with_labels(y, prob_fake)
print("acc,prec,rec,f1:", acc,prec,rec,f1)
print("cm:\n",cm, "auc:", auc)

print("\n--- Using swapped labels (1<->0) ---")
y_swapped = 1 - y
acc2,prec2,rec2,f12,cm2,auc2 = eval_with_labels(y_swapped, prob_fake)
print("acc,prec,rec,f1:", acc2,prec2,rec2,f12)
print("cm:\n",cm2, "auc:", auc2)