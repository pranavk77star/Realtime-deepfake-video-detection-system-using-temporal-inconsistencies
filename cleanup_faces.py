# cleanup_faces.py
import os
import sys
import cv2

base = r".\datasets\faces"   # adjust if you used other folder
bad = 0
for root, _, files in os.walk(base):
    for f in files:
        if not f.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        p = os.path.join(root, f)
        try:
            if os.path.getsize(p) == 0:
                os.remove(p); bad += 1; print("Removed zero-byte:", p); continue
            im = cv2.imread(p)
            if im is None:
                os.remove(p); bad += 1; print("Removed unreadable:", p)
        except Exception as e:
            print("Error checking", p, e)
print("Done. Removed:", bad)
