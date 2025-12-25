# analyze_preds.py
import sys, numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_curve

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print("NPZ keys:", keys)

    # direct mappings for your file
    if 'y_all' in d and 's_all' in d:
        return d['y_all'], d['s_all']

    # common variants
    if 'y_true' in d and 'y_score' in d:
        return d['y_true'], d['y_score']

    # fallback guesses
    y_key = next((k for k in keys if any(x in k.lower() for x in ['y', 'label', 'true'])), None)
    s_key = next((k for k in keys if any(x in k.lower() for x in ['score', 'pred', 's'])), None)
    if not y_key or not s_key:
        raise RuntimeError(f"Couldn't find label/score arrays in npz keys: {keys}")
    return d[y_key], d[s_key]

def print_stats(y_true, y_score, threshold):
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score).astype(float)
    n = len(y_true)
    print("N:", n)
    print("mean score REAL (y==0):", y_score[y_true==0].mean() if (y_true==0).any() else None)
    print("mean score FAKE (y==1):", y_score[y_true==1].mean() if (y_true==1).any() else None)
    print("min,max:", y_score.min(), y_score.max())
    print("frac > threshold:", (y_score > threshold).mean())
    # predictions at threshold
    y_pred = (y_score >= threshold).astype(int)
    print("\nThreshold = %.4f" % threshold)
    print("Accuracy:", (y_pred == y_true).mean())
    print("Confusion matrix (rows=true Real,Fake):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    return y_pred

def find_best_threshold(y_true, y_score):
    # search thresholds that maximize F1 (dense)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of len = len(precisions)-1
    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    return best_thr, best_f1

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_preds.py <preds.npz> [threshold]")
        sys.exit(1)
    path = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.5
    y_true, y_score = load_npz(path)
    # some dump scripts store scores for 'Fake' class (1) already (we used --invert earlier)
    # we only use values as-is.
    print("Loaded:", path)
    try:
        _ = y_true.shape
    except Exception as e:
        y_true = np.array(y_true)
    try:
        _ = y_score.shape
    except Exception as e:
        y_score = np.array(y_score)
    # print basic stats + confusion at provided threshold
    print_stats(y_true, y_score, thresh)
    # find best threshold by F1
    best_thr, best_f1 = find_best_threshold(y_true, y_score)
    print("\nBest F1 on this set = %.4f at threshold = %.4f" % (best_f1, best_thr))
    # show confusion at best threshold
    print("\n--- At best threshold ---")
    print_stats(y_true, y_score, best_thr)

if __name__ == "__main__":
    main()