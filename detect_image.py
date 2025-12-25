# detect_image.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback
import pickle
import time

def swish(x):
    return tf.nn.swish(x)

_MODEL_CACHE = {}
_CALIB_CACHE = None

def load_model_cached(model_path):
    global _MODEL_CACHE
    model_path = os.path.abspath(model_path)
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"swish": swish})
    except Exception:
        model = tf.keras.models.load_model(model_path)
    _MODEL_CACHE[model_path] = model
    print(f"[INFO] Model loaded from: {model_path}")
    return model

def load_calibrator(cal_path="prob_calibrator.pkl"):
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "rb") as f:
                _CALIB_CACHE = pickle.load(f)
            print(f"[INFO] Calibrator loaded: {cal_path}")
        except Exception as e:
            print("[WARN] Failed to load calibrator:", e)
            _CALIB_CACHE = None
    return _CALIB_CACHE

def preprocess_pil(img_pil, img_size):
    img = img_pil.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def _safe_sigmoid(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-arr))

def interpret_output(pred, invert=False):
    """
    Return (raw, prob_fake, prob_real).
    invert: swap meaning of model outputs (index mapping), do NOT compute 1 - p.
    """
    try:
        pred = np.asarray(pred)
        # two-output (softmax/logits)
        if pred.ndim == 2 and pred.shape[1] >= 2:
            row = pred[0].astype(np.float64)
            s = float(np.sum(row))
            if abs(s - 1.0) < 1e-3 and np.all(row >= 0.0):
                prob_fake = float(row[1])
                prob_real = float(row[0])
            else:
                ex = np.exp(row - np.max(row))
                soft = ex / np.sum(ex)
                prob_fake = float(soft[1])
                prob_real = float(soft[0])
            raw = prob_fake
            if invert:
                prob_fake, prob_real = prob_real, prob_fake
                raw = prob_fake
            return raw, float(prob_fake), float(prob_real)

        # single output (sigmoid/logit)
        flat = np.ravel(pred)
        if flat.size == 0:
            raise ValueError("Empty prediction")
        s = float(flat[0])
        if s < 0.0 or s > 1.0:
            prob_fake = float(_safe_sigmoid(s))
            prob_real = 1.0 - prob_fake
        else:
            prob_fake = s
            prob_real = 1.0 - s
        raw = prob_fake
        if invert:
            prob_fake, prob_real = prob_real, prob_fake
            raw = prob_fake
        return raw, float(prob_fake), float(prob_real)
    except Exception as e:
        print("[interpret_output] failed:", e)
        traceback.print_exc()
        return 0.0, 0.0, 1.0

def detect_from_path(
    path,
    model_path="outputs/fine_tuned_v3.keras",
    threshold=0.5,
    invert=False,
    img_size=128,
    smooth=False,
    window_size=3
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    model = load_model_cached(model_path)
    cal = load_calibrator()

    pil = Image.open(path)
    inp = preprocess_pil(pil, img_size)
    pred = model.predict(inp, verbose=0)
    raw, prob_fake, prob_real = interpret_output(pred, invert=invert)

    # calibrator (if present)
    if cal is not None:
        try:
            if hasattr(cal, "predict_proba"):
                prob_fake = float(cal.predict_proba([[prob_fake]])[:, 1])
            else:
                prob_fake = float(cal.predict([prob_fake])[0])
            prob_fake = max(0.0, min(1.0, prob_fake))
            prob_real = 1.0 - prob_fake
        except Exception as e:
            print("[WARN] Calibration failed:", e)

    # smoothing (optional)
    if smooth:
        global _SMOOTH_HISTORY
        if "_SMOOTH_HISTORY" not in globals():
            _SMOOTH_HISTORY = []
        _SMOOTH_HISTORY.append(prob_fake)
        if len(_SMOOTH_HISTORY) > window_size:
            _SMOOTH_HISTORY.pop(0)
        prob_fake = float(np.mean(_SMOOTH_HISTORY))
        prob_real = 1.0 - prob_fake

    label = "Fake" if prob_fake >= threshold else "Real"
    confidence = max(prob_fake, prob_real)

    return label, float(confidence), float(raw), float(prob_fake), float(prob_real)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--invert", action="store_true", default=False)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--smooth", action="store_true", default=False)
    args = parser.parse_args()

    t0 = time.time()
    label, conf, raw, pf, pr = detect_from_path(
        args.img,
        model_path=args.model,
        threshold=args.threshold,
        invert=args.invert,
        img_size=args.img_size,
        smooth=args.smooth
    )
    dt = time.time() - t0

    print("Image:", args.img)
    print("Inference time:", f"{dt:.3f}s")
    print("raw model score:", f"{raw:.6f}")
    print("prob_fake:", f"{pf:.6f}", "prob_real:", f"{pr:.6f}")
    print(f"Threshold: {args.threshold} -> Result: {label} (confidence {conf:.2f})")