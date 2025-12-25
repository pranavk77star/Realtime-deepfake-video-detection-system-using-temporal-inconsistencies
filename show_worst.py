#!/usr/bin/env python3
# detect_image.py - single-image prediction (consistent with balanced model defaults)

import argparse
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow.keras.activations import swish

parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="Path to input image")
parser.add_argument("--model", type=str, default="outputs/final_best_model_balanced.keras")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--threshold", type=float, default=0.9980)
parser.add_argument("--invert", action="store_true", default=True,
                    help="Invert model output (use when model outputs high=REAL). Default: True")
args = parser.parse_args()

MODEL_PATH = args.model
IMG_SIZE = args.img_size
THRESH = args.threshold
INVERT = True if args.invert else False

print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"swish": swish})
print("Model loaded.")

def _preprocess_image(img: np.ndarray, size=(IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def detect_fake_from_array(img: np.ndarray, threshold: float = THRESH, invert: bool = INVERT) -> Tuple[str, float]:
    x = _preprocess_image(img)
    raw = float(model.predict(x, verbose=0).ravel()[0])
    # invert because your model gives higher for REAL
    score = 1.0 - raw if invert else raw
    if score >= threshold:
        label = "Fake"
        confidence = score
    else:
        label = "Real"
        confidence = 1.0 - score
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence

def detect_fake(image_path: str, threshold: float = THRESH, invert: bool = INVERT) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    label, conf = detect_fake_from_array(img, threshold=threshold, invert=invert)
    return f"{label} (confidence {conf:.2f})"

if __name__ == "__main__":
    res = detect_fake(args.image, threshold=THRESH, invert=INVERT)
    print(res)