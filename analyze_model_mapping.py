# analyze_model_mapping.py
import os, pickle, numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

VAL_DIR = "datasets/balanced/val"
MODEL = "outputs/fine_tuned_v3.keras"
CAL_PATH = "prob_calibrator.pkl"

def load_model(path):
    m = tf.keras.models.load_model(path, compile=False)
    print("[INFO] model loaded:", path)
    return m

def load_cal(path):
    if os.path.exists(path):
        try:
            with open(path,"rb") as f:
                c = pickle.load(f)
            print("[INFO] calibrator loaded:", path, "type:", type(c))
            return c
        except Exception as e:
            print("[WARN] failed load calibrator:", e)
    else:
        print("[INFO] calibrator not found:", path)
    return None

def get_raw_prob(pred_arr):
    # returns scalar prob from model output array
    pred = np.asarray(pred_arr)
    if pred.ndim == 2 and pred.shape[1] >= 2:
        row = pred[0]
        s = float(np.sum(row))
        if abs(s - 1.0) < 1e-3 and np.all(row >= 0.0):
            return float(row[1])
        else:
            ex = np.exp(row - np.max(row))
            soft = ex / np.sum(ex)
            return float(soft[1])
    flat = np.ravel(pred)
    if flat.size == 0:
        return 0.0
    s = float(flat[0])
    if s < 0.0 or s > 1.0:
        return 1.0 / (1.0 + np.exp(-s))
    return s

def apply_cal(cal, p):
    if cal is None: return p
    try:
        if hasattr(cal, "predict_proba"):
            return float(cal.predict_proba([[p]])[:,1])
        else:
            return float(cal.predict([p])[0])
    except Exception as e:
        print("  [WARN] calibrator call failed:", e)
        return p

def main():
    gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR, target_size=(128,128), batch_size=32, class_mode="binary", shuffle=False)
    model = load_model(MODEL)
    cal = load_cal(CAL_PATH)

    xs, ys = [], []
    # we will go through generator once
    for xb, yb in gen:
        xs.append(xb)
        ys.append(yb)
        if len(xs)*xb.shape[0] >= gen.samples:
            break
    X = np.vstack(xs)
    Y = np.concatenate(ys).astype(int)
    print("Loaded X shape:", X.shape, "Y shape:", Y.shape)

    raws = []
    batch = 64
    for i in range(0, X.shape[0], batch):
        preds = model.predict(X[i:i+batch], verbose=0)
        for p in preds:
            raws.append(get_raw_prob(p))
    raws = np.array(raws)
    assert raws.shape[0] == Y.shape[0]

    for invert in (False, True):
        print("\n--- invert =", invert, "---")
        probs = []
        for r in raws:
            if invert:
                mapped = 1.0 - r
            else:
                mapped = r
            calibrated = apply_cal(cal, mapped)
            probs.append(calibrated)
        probs = np.array(probs)

        for cls_idx, cls_name in sorted(gen.class_indices.items(), key=lambda x:x[1]):
            mask = (Y == cls_idx)
            print(f"class {cls_name} (index {cls_idx}): count={mask.sum()}")
            print(" raw mean {:.4f} median {:.4f} std {:.4f}".format(raws[mask].mean(), np.median(raws[mask]), raws[mask].std()))
            print(" mapped mean {:.4f} median {:.4f}".format((1.0 - raws[mask]).mean() if invert else raws[mask].mean(), (1.0 - np.median(raws[mask])) if invert else np.median(raws[mask])))
            print(" calibrated mean {:.4f} median {:.4f}".format(probs[mask].mean(), np.median(probs[mask])))
        # show few sample pairs
        print(" sample (true, raw, mapped, calib) first 10:")
        for i in range(10):
            r = raws[i]
            mapped = 1.0 - r if invert else r
            c = apply_cal(cal, mapped)
            print(i, int(Y[i]), f"{r:.6f}", f"{mapped:.6f}", f"{c:.6f}")

if __name__ == "__main__":
    main()