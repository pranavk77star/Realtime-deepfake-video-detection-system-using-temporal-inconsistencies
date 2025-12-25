# diagnose_dataset_and_model.py
import os
import sys
import csv
from collections import Counter
from datetime import datetime

def count_folders(path):
    counts = {}
    if not os.path.exists(path):
        return None
    for cls in sorted(os.listdir(path)):
        cls_p = os.path.join(path, cls)
        if os.path.isdir(cls_p):
            counts[cls] = sum(1 for _ in os.listdir(cls_p) if os.path.isfile(os.path.join(cls_p, _)))
    return counts

def count_from_csv(csv_path, label_col="label", file_col="filename"):
    if not os.path.exists(csv_path):
        return None
    cnt = Counter()
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if label_col not in reader.fieldnames:
                # try common alternatives
                for alt in ["label","class","target"]:
                    if alt in reader.fieldnames:
                        label_col = alt
                        break
            for r in reader:
                if label_col in r and r[label_col] != "":
                    cnt[r[label_col]] += 1
    except Exception as e:
        return {"error": str(e)}
    return dict(cnt)

def print_counts(title, d):
    if d is None:
        print(f"{title}: not found")
        return
    total = sum(d.values())
    print(f"\n{title} (total = {total}):")
    for k,v in d.items():
        pct = 100.0 * v / total if total>0 else 0.0
        print(f"  {k}: {v} ({pct:.2f}%)")
    # show simple imbalance warning
    if total>0:
        most = max(d.values())
        least = min(d.values())
        if most / max(1, least) > 3:
            print("  >> WARNING: dataset appears imbalanced (major class >3x minor). Consider balancing/weights/oversampling.")

def list_examples(folder, max_examples=3):
    if not os.path.exists(folder):
        return
    print(f"\nExample files in {folder}:")
    for cls in sorted(os.listdir(folder)):
        cls_p = os.path.join(folder, cls)
        if os.path.isdir(cls_p):
            files = [f for f in os.listdir(cls_p) if os.path.isfile(os.path.join(cls_p, f))]
            print(f"  {cls}: {len(files)} files, examples: {files[:max_examples]}")

def check_model(path="model.pth"):
    if not os.path.exists(path):
        print("\nModel file model.pth: NOT FOUND")
        return
    size = os.path.getsize(path)
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nModel file: {path}")
    print(f"  size: {size/1e6:.2f} MB")
    print(f"  modified: {mtime}")
    # quick sanity: is it a torch zip file (contains PK) or plain?
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
            if head.startswith(b"PK\x03\x04"):
                print("  format: zip / possibly PyTorch saved with zipfile container (torch.save).")
            else:
                print("  format: unknown binary header -> probably OK (torch).")
    except Exception as e:
        print("  couldn't read file header:", e)

def main():
    print("Running quick diagnostics...\n(Assumes folders like data/train/<class> and data/val/<class>)")
    train_counts = count_folders("data/train")
    val_counts = count_folders("data/val")
    csv_counts = count_from_csv("labels.csv")  # optional CSV
    print_counts("TRAIN", train_counts)
    print_counts("VAL", val_counts)
    if csv_counts is not None:
        print_counts("labels.csv", csv_counts)
    # list some example file names
    if train_counts:
        list_examples("data/train")
    if val_counts:
        list_examples("data/val")
    # check model file
    check_model("model.pth")

    print("\nDone. Copy & paste the output here if you want me to interpret it and give exact next commands.\n")

if __name__ == "__main__":
    main()