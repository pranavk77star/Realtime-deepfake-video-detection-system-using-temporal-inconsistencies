#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate a trained binary classification Keras model on a directory of test images.

Saves:
 - <outprefix>_confusion.png          (confusion at chosen threshold)
 - <outprefix>_confusion_0.5.png      (optional baseline at 0.5)
 - <outprefix>_roc.png
 - <outprefix>_classification_report.txt
 - <outprefix>_predictions.csv
 - <outprefix>_predictions.npz
"""
import os
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import swish
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve

# non-interactive backend (safe for headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional seaborn
try:
    import seaborn as sns  # noqa: F401
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# ---------- custom dropout wrapper in case model used it ----------
@tf.keras.utils.register_keras_serializable()
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Keras model on a test folder.")
    p.add_argument("--model", required=True, help="Path to .keras model file (checkpoint/best model).")
    p.add_argument("--test_dir", required=True, help="Test directory (folders per class).")
    p.add_argument("--img_size", type=int, default=None,
                   help="Image size (square). If omitted, try to auto-detect from the model.")
    p.add_argument("--batch", type=int, default=32, help="Batch size for prediction.")
    p.add_argument("--out", required=False, default="reports/eval",
                   help="Output path prefix for saved reports (folder/nameprefix).")
    p.add_argument("--threshold", default="0.5",
                   help="Decision threshold for positive class. Use 'auto' to pick best threshold from ROC (Youden). Default 0.5")
    p.add_argument("--no_baseline_png", dest="baseline_png", action="store_false",
                   help="Don't save baseline confusion matrix at threshold 0.5 when using --threshold auto.")
    return p.parse_args()


def ensure_out_dir(prefix):
    out_dir = os.path.dirname(prefix) or "reports"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_get_img_size(model, provided):
    if provided is not None:
        return int(provided)
    try:
        inp = model.input_shape
        if isinstance(inp, list):
            inp = inp[0]
        h = inp[1]
        w = inp[2]
        if h is None or w is None:
            raise ValueError("model input_shape has None spatial dims")
        if h != w:
            print(f"[Warn] model input is not square ({h}x{w}) — using height {h} for resize.")
        print(f"[Auto] Using model input size: {h}x{h}")
        return int(h)
    except Exception as e:
        raise RuntimeError("Could not auto-detect input size; provide --img_size") from e


def normalize_model_output(y_raw):
    """
    Convert model outputs to a single probability per sample for class index 1.
    Handles:
     - shape (N, 1) -> ravel()
     - shape (N, 2) -> softmax-like: take column 1 (assumed class-index mapping matches)
     - shape (N,) -> return as is
     - logits (values outside [0,1]) -> attempt sigmoid or softmax
    """
    y = np.asarray(y_raw)
    # common cases
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    if y.ndim == 2 and y.shape[1] == 2:
        # assume columns correspond to class indices used by flow_from_directory.
        # column 1 => probability of index 1
        col1 = y[:, 1]
        # if values look like logits (many <0 or >1) try softmax
        if (col1.min() < 0) or (col1.max() > 1):
            try:
                import scipy.special as sc
                soft = sc.softmax(y, axis=1)[:, 1]
                return soft
            except Exception:
                # fallback to dividing by sum (naive)
                s = np.exp(y)
                ssum = s.sum(axis=1, keepdims=True)
                return (s[:, 1] / ssum[:, 0]).ravel()
        return col1
    if y.ndim == 1:
        # if values not in [0,1], assume logits and apply sigmoid
        if (y.min() < 0) or (y.max() > 1):
            try:
                return 1.0 / (1.0 + np.exp(-y))
            except Exception:
                # as last resort normalize via min-max to [0,1]
                ym = (y - y.min()) / (y.max() - y.min() + 1e-12)
                return ym
        return y
    # other shapes: flatten and try to interpret
    flat = y.ravel()
    if (flat.min() < 0) or (flat.max() > 1):
        try:
            return 1.0 / (1.0 + np.exp(-flat))
        except Exception:
            ym = (flat - flat.min()) / (flat.max() - flat.min() + 1e-12)
            return ym
    return flat


def plot_confusion(cm, labels, outpath, title="Confusion matrix"):
    plt.figure(figsize=(5, 4))
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center", color="white" if cm[i, j] > cm.max()/2. else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved confusion matrix to", outpath)


def main():
    args = parse_args()
    OUT_PREFIX = args.out
    OUT_DIR = ensure_out_dir(OUT_PREFIX)

    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model, custom_objects={"FixedDropout": FixedDropout, "swish": swish})

    IMG_SIZE = safe_get_img_size(model, args.img_size)
    BATCH_SIZE = int(args.batch)
    TEST_DIR = args.test_dir

    print(f"Preparing test generator from: {TEST_DIR}  (batch={BATCH_SIZE}, size={IMG_SIZE})")
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    steps = math.ceil(test_gen.samples / float(test_gen.batch_size))
    print(f"Predicting {test_gen.samples} images (batch={BATCH_SIZE}, steps={steps}) ...")
    y_raw = model.predict(test_gen, steps=steps, verbose=1)

    # normalise to probabilities for class index 1
    y_prob = normalize_model_output(y_raw)

    # trim/pad to test_gen.samples if necessary
    if len(y_prob) != test_gen.samples:
        print(f"[Warn] prediction length {len(y_prob)} != samples {test_gen.samples}. Adjusting...")
        y_prob = y_prob[:test_gen.samples]

    y_true = test_gen.classes[: len(y_prob)]

    # build label name mapping such that label_names[index] = folder name
    class_indices = test_gen.class_indices
    num_classes = max(class_indices.values()) + 1
    label_names = [None] * num_classes
    for name, idx in class_indices.items():
        if 0 <= idx < num_classes:
            label_names[idx] = name
    label_names = [ln if ln is not None else f"cls{idx}" for idx, ln in enumerate(label_names)]

    # determine threshold
    chosen_threshold = None
    if args.threshold.lower() == "auto":
        # compute ROC and choose threshold that maximizes tpr - fpr (Youden's J)
        if len(np.unique(y_true)) != 2:
            raise RuntimeError("Auto threshold requires binary classification with 2 classes.")
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        chosen_threshold = float(thr[best_idx])
        print(f"[Auto] Chosen threshold by Youden's J: {chosen_threshold:.4f}")
    else:
        chosen_threshold = float(args.threshold)
        print(f"Using fixed threshold: {chosen_threshold}")

    y_pred = (y_prob > chosen_threshold).astype(int)

    # confusion matrix and plots
    cm = confusion_matrix(y_true, y_pred)
    cm_path = f"{OUT_PREFIX}_confusion.png"
    plot_confusion(cm, label_names, cm_path, title=f"Confusion Matrix (thr={chosen_threshold:.3f})")

    # if auto and user wants baseline, also save confusion at 0.5
    if args.threshold.lower() == "auto" and args.baseline_png:
        baseline_pred = (y_prob > 0.5).astype(int)
        cm0 = confusion_matrix(y_true, baseline_pred)
        cm0_path = f"{OUT_PREFIX}_confusion_0.5.png"
        plot_confusion(cm0, label_names, cm0_path, title="Confusion Matrix (thr=0.5)")

    # classification report
    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    txt_path = f"{OUT_PREFIX}_classification_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print("Saved classification report to", txt_path)
    print("\nClassification Report:\n", report)

    # ROC / AUC
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], "--", linewidth=0.7)
            # annotate chosen threshold on ROC if available
            if chosen_threshold is not None:
                # find closest thr index
                idx = np.argmin(np.abs(thr - chosen_threshold))
                plt.scatter(fpr[idx], tpr[idx], c='r', s=40, label=f"thr {chosen_threshold:.3f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            roc_path = f"{OUT_PREFIX}_roc.png"
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()
            print(f"Saved ROC curve to {roc_path} (AUC={auc:.4f})")
        else:
            print("ROC/AUC skipped: not a binary classification (need exactly 2 classes).")
    except Exception as e:
        print("ROC could not be computed:", e)

    # optionally save precision-recall curve
    try:
        if len(np.unique(y_true)) == 2:
            prec, rec, pthr = precision_recall_curve(y_true, y_prob)
            plt.figure()
            plt.plot(rec, prec, label="Precision-Recall")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.tight_layout()
            pr_path = f"{OUT_PREFIX}_pr.png"
            plt.savefig(pr_path)
            plt.close()
            print("Saved Precision-Recall curve to", pr_path)
    except Exception:
        pass

    # save predictions CSV/NPZ
    filenames = np.array(test_gen.filenames)[: len(y_prob)]
    df = pd.DataFrame({
        "filename": filenames,
        "true_label_index": y_true,
        "true_label_name": [label_names[i] for i in y_true],
        "pred_prob": y_prob,
        "pred_label_index": y_pred,
        "pred_label_name": [label_names[i] for i in y_pred],
    })
    csv_path = f"{OUT_PREFIX}_predictions.csv"
    df.to_csv(csv_path, index=False)
    npz_path = f"{OUT_PREFIX}_predictions.npz"
    np.savez(npz_path, y_true=y_true, y_prob=y_prob, y_pred=y_pred, filenames=filenames)
    print("Saved predictions to", csv_path, "and", npz_path)

    print("Done.")


if __name__ == "__main__":
    main()

