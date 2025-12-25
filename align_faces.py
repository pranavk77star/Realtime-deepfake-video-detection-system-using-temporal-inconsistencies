# tools/align_faces.py
import os,glob,cv2
import mediapipe as mp
import numpy as np
import shutil,sys

src_root = sys.argv[1]  # e.g. datasets/faces
dst_root = sys.argv[2]  # e.g. datasets/aligned_faces
PAD = float(sys.argv[3]) if len(sys.argv)>3 else 0.25
os.makedirs(dst_root, exist_ok=True)

mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)

for label in ("real","Real","fake","Fake","FF_original","FF_Deepfakes"):
    # detect case-insensitively
    for folder in glob.glob(os.path.join(src_root, label+"*")):
        if not os.path.isdir(folder): continue
        out_folder = os.path.join(dst_root, os.path.basename(folder))
        os.makedirs(out_folder, exist_ok=True)
        for imgpath in glob.glob(os.path.join(folder,"**","*.png"), recursive=True)+glob.glob(os.path.join(folder,"**","*.jpg"), recursive=True):
            fname = os.path.basename(imgpath)
            outpath = os.path.join(out_folder, fname)
            if os.path.exists(outpath): continue
            img = cv2.imread(imgpath)
            if img is None: continue
            H,W = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = mp_fd.process(rgb)
            if not res or not res.detections:
                continue
            det = res.detections[0]
            bb = det.location_data.relative_bounding_box
            x1 = max(0,int(bb.xmin*W)); y1 = max(0,int(bb.ymin*H))
            w = max(1,int(bb.width*W)); h = max(1,int(bb.height*H))
            padw = int(w*PAD); padh = int(h*PAD)
            x1p = max(0,x1-padw//2); y1p = max(0,y1-padh//2)
            x2p = min(W,x1+w+padw//2); y2p = min(H,y1+h+padh//2)
            face = img[y1p:y2p, x1p:x2p]
            face = cv2.resize(face, (128,128))
            cv2.imwrite(outpath, face)
print("Done.")