# check_accuracy.py
import os
import argparse
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# reuse the same loading helpers from your repo if present
# this will allow swish/custom objects handling
try:
    from detect_image import swish, load_model_cached
    _USE_DETECT_IMAGE = True
except Exception:
    _USE_DETECT_IMAGE = False

def load_keras_model(path):
    path = os.path.abspath(path)
    if _USE_DETECT_IMAGE:
        # prefer using your project's loader which handles custom_objects and caching
        return load_model_cached(path)
    # fallback: standard load_model
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        # try with swish custom object if available in this env
        try:
            return tf.keras.models.load_model(path, custom_objects={"swish": tf.nn.swish}, compile=False)
        except Exception:
            raise e

def preds_to_prob_fake(preds, invert=False):
    """
    Convert raw model preds -> prob_fake array (shape = (N,))
    Supports:
      - logits/probabilities shape (N,2) -> treat index 1 as fake
      - single-output shape (N,) or (N,1) -> treat as sigmoid: s = prob_fake (unless invert True)
    """
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        # multi-class softmax-like output: assume index 1 = fake
        prob_fake = preds[:, 1].astype(float)
        return prob_fake
    # single-value outputs
    flat = preds.ravel().astype(float)
    if invert:
        # single output represents prob_real -> convert
        return (1.0 - flat)
    else:
        # single output represents prob_fake
        return flat

def main():
    p = argparse.ArgumentParser(description="Evaluate Keras deepfake model on a validation folder")
    p.add_argument("--model", required=True, help="Path to .keras model file")
    p.add_argument("--val_dir", required=True, help="Validation directory in flow_from_directory format (subfolders per class)")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--invert", action="store_true", help="If model single-output is prob_real instead of prob_fake, set --invert")
    p.add_argument("--threshold", type=float, default=None, help="If provided, use this threshold to compute predicted labels (default 0.5)")
    args = p.parse_args()

    model_path = args.model
    val_dir = args.val_dir
    img_size = args.img_size
    batch_size = args.batch_size
    invert = args.invert
    threshold = args.threshold if args.threshold is not None else 0.5

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation dir not found: {val_dir}")

    print("Loading model:", model_path)
    model = load_keras_model(model_path)
    # compile not required, but some Keras functions expect metrics etc; we'll not rely on compile
    print("Model loaded.")

    # Setup image generator (same preprocessing as detect_image: rescale 1/255)
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",   # expects two-class directory
        shuffle=False
    )

    # Info about class mapping
    print("Class indices mapping (folder -> index):", val_gen.class_indices)
    # val_gen.classes is numpy array of labels in generator order (0..n-1)
    n_samples = val_gen.samples
    steps = int(math.ceil(n_samples / batch_size))
    print(f"Found {n_samples} images, batch_size={batch_size}, steps={steps}")

    # Predict on the generator
    print("Running model.predict on validation set ...")
    preds = model.predict(val_gen, steps=steps, verbose=1)
    # preds shape may be (N,1), (N,), (N,2), etc.
    prob_fake = preds_to_prob_fake(preds, invert=invert)

    # Ensure prob_fake length matches val_gen.samples
    if prob_fake.shape[0] > n_samples:
        prob_fake = prob_fake[:n_samples]
    elif prob_fake.shape[0] < n_samples:
        raise RuntimeError("Predictions fewer than expected samples: got %d expected %d" % (prob_fake.shape[0], n_samples))

    y_true = val_gen.classes  # 0/1 labels assigned by flow_from_directory
    # We assume class index 1 corresponds to "fake". If your folder names produce opposite mapping, user should check mapping printed above.
    # Build predicted binary labels using threshold
    y_pred = (prob_fake >= threshold).astype(int)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc = None
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, prob_fake)
    except Exception:
        auc = None

    print("\n=== EVALUATION RESULTS ===")
    print(f"Samples: {n_samples}")
    print("Threshold used for prediction:", threshold)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix (rows = true [0=class0,1=class1], cols = pred):")
    print(cm)
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: could not compute")

    # Show a short sample of predictions for debugging
    print("\nExample predictions (first 10):")
    for i in range(min(10, n_samples)):
        print(f"{i:03d}: true={y_true[i]} prob_fake={prob_fake[i]:.6f} -> pred={y_pred[i]}")

    # Done
    print("\nDone.")

if __name__ == "__main__":
    main()