# 00-check_dataset.py
import os

split_path = "./split_dataset"
subfolders = ["Celeb", "FF"]

for dataset in subfolders:
    print(f"\n=== Dataset: {dataset} ===")
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            cls_dir = os.path.join(split_path, dataset, split, cls)
            if not os.path.exists(cls_dir):
                print(f"Missing folder: {cls_dir}")
                continue
            n_files = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg'))])
            print(f"{split}/{cls}: {n_files} images")
