# tools/make_balanced_splits.py
import os,glob,random,shutil,sys
src = sys.argv[1]  # aligned_faces root
dst = sys.argv[2]  # balanced dataset root
os.makedirs(dst, exist_ok=True)
# gather all real & all fake
realfolders = [p for p in glob.glob(os.path.join(src,"**")) if os.path.isdir(p) and 'real' in os.path.basename(p).lower()]
fakefolders = [p for p in glob.glob(os.path.join(src,"**")) if os.path.isdir(p) and 'fake' in os.path.basename(p).lower()]

real_files=[]
fake_files=[]
for f in realfolders:
    real_files += glob.glob(os.path.join(f,"*.png")) + glob.glob(os.path.join(f,"*.jpg"))
for f in fakefolders:
    fake_files += glob.glob(os.path.join(f,"*.png")) + glob.glob(os.path.join(f,"*.jpg"))

random.shuffle(real_files); random.shuffle(fake_files)
N = min(len(real_files), len(fake_files))
real_files = real_files[:N]
fake_files = fake_files[:N]
print("Using",N,"per class")

# make splits 80/10/10
def split_and_copy(files, label):
    n = len(files)
    t1 = int(n*0.8); t2 = t1 + int(n*0.1)
    parts = {"train":files[:t1], "val":files[t1:t2], "test":files[t2:]}
    for part, fls in parts.items():
        out = os.path.join(dst, part, label)
        os.makedirs(out, exist_ok=True)
        for srcf in fls:
            dest = os.path.join(out, os.path.basename(srcf))
            shutil.copy(srcf, dest)

split_and_copy(real_files, "Real")
split_and_copy(fake_files, "Fake")
print("Balanced dataset ready at", dst)