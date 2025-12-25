import os, shutil, random, sys, glob

def sample(src, dest, n=400):
    os.makedirs(dest, exist_ok=True)
    imgs = glob.glob(os.path.join(src, "**", "*.jpg"), recursive=True)
    random.shuffle(imgs)
    for i, path in enumerate(imgs[:n]):
        shutil.copy(path, os.path.join(dest, f"real_{i:04d}.jpg"))
    print(f"✅ Copied {min(n, len(imgs))} real images to {dest}")

if __name__ == "__main__":
    src = sys.argv[1]
    dest = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    sample(src, dest, n)