# tools/dedupe.py
import os,hashlib,shutil,sys
root = sys.argv[1]  # e.g. datasets/faces
seen = {}
for sub, _, files in os.walk(root):
    for f in files:
        if not f.lower().endswith((".jpg",".jpeg",".png")): continue
        path = os.path.join(sub,f)
        h = hashlib.md5(open(path,'rb').read()).hexdigest()
        if h in seen:
            # move duplicate to a trash folder
            trash = os.path.join(root, "_dupes")
            os.makedirs(trash, exist_ok=True)
            new = os.path.join(trash, os.path.basename(path))
            print("Moving duplicate:", path)
            shutil.move(path, new)
        else:
            seen[h]=path
print("Done.")