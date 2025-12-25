# check_preds.py
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "outputs/final_best_model.keras"   # change if different
IMG_PATH = sys.argv[1] if len(sys.argv)>1 else "test_fake.jpg"  # put path to a known fake image
IMG_SIZE = (128,128)   # put the size you used during training (very important)

model = load_model(MODEL_PATH)
print("Model loaded:", MODEL_PATH)

img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit("Couldn't open image: " + IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = img.astype("float32") / 255.0   # adjust if you trained differently
x = np.expand_dims(img, axis=0)

raw = model.predict(x)    # shape may be (1,1) or (1,2)
print("Raw prediction output:", raw)

# Interpret depending on shape:
if raw.ndim==2 and raw.shape[1]==1:
    score = float(raw[0,0])   # sigmoid output typical
    print("Sigmoid score (0..1). Higher -> class 1")
    print(f"score = {score:.4f}")
    # choose threshold here
    thresh = 0.5
    label = "Fake" if score > thresh else "Real"
    print("Label @ threshold", thresh, "->", label)
elif raw.ndim==2 and raw.shape[1]==2:
    probs = np.squeeze(raw)
    print("Softmax probs (class0, class1) =", probs)
    # If you know class index: class1 might be fake
    if probs[1] > probs[0]:
        print("Label by argmax -> Class1 (prob {:.3f})".format(probs[1]))
    else:
        print("Label by argmax -> Class0 (prob {:.3f})".format(probs[0]))
else:
    print("Unexpected output shape:", raw.shape)