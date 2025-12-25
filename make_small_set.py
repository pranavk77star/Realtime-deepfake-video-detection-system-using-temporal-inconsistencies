# make_small_set.py
import os, random, shutil
random.seed(0)

src_base = "split_dataset_final"
dst_base = "tmp_small"

for split, n in (("train", 40), ("val", 20)):
    for cls in ("real","fake"):
        src = os.path.join(src_base, split, cls)
        dst = os.path.join(dst_base, split, cls)
        os.makedirs(dst, exist_ok=True)
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src,f))]
        pick = random.sample(files, min(len(files), n//2 if split=="train" else n//2))
        for f in pick:
            shutil.copy(os.path.join(src,f), os.path.join(dst,f))
print("tmp_small created: tmp_small/train and tmp_small/val")

