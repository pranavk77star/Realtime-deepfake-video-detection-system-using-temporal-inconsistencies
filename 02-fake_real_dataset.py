# 02-create_splits_from_faces.py
import os, sys, shutil
import splitfolders

if len(sys.argv) < 4:
    print("Usage: python 02-create_splits_from_faces.py <real_faces_root> <fake_faces_root> <out_root> [train=0.8 val=0.1 test=0.1]")
    sys.exit(1)

real_root = sys.argv[1]
fake_root = sys.argv[2]
out_root = sys.argv[3]
train = float(sys.argv[4]) if len(sys.argv)>4 else 0.8
val = float(sys.argv[5]) if len(sys.argv)>5 else 0.1
test = float(sys.argv[6]) if len(sys.argv)>6 else 0.1

assert abs(train+val+test - 1.0) < 1e-6, "Splits must add to 1.0"

tmp_dataset = os.path.join(out_root, "temp_dataset")
os.makedirs(tmp_dataset, exist_ok=True)

real_dst = os.path.join(tmp_dataset, "real")
fake_dst = os.path.join(tmp_dataset, "fake")
os.makedirs(real_dst, exist_ok=True)
os.makedirs(fake_dst, exist_ok=True)

def copy_files(src_dir, dst_dir):
    for root,_,files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                src = os.path.join(root,f)
                dst = os.path.join(dst_dir, f"{os.path.splitext(f)[0]}_{abs(hash(src))%100000}.png")
                shutil.copyfile(src, dst)

print("Copying real faces...")
copy_files(real_root, real_dst)
print("Copying fake faces...")
copy_files(fake_root, fake_dst)

print("Running splitfolders...")
splitfolders.ratio(tmp_dataset, output=out_root, seed=123, ratio=(train, val, test))
print("Done. Splits created at:", out_root)
