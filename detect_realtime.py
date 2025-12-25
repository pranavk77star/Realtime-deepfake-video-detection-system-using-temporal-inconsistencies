# detect_realtime.py
import os
import time
import argparse
from collections import deque
import cv2
import numpy as np
from detect_image import load_model_cached, preprocess_pil, interpret_output, load_calibrator
from PIL import Image

_use_mtcnn = False
detector = None
try:
    from mtcnn import MTCNN
    detector = MTCNN()
    _use_mtcnn = True
    print("[detect_realtime] Using MTCNN detector")
except Exception:
    _use_mtcnn = False
    detector = None
    print("[detect_realtime] MTCNN not available; using full-frame fallback")

def predict_frame_array(img_bgr, model, img_size=128, invert=False):
    h, w = img_bgr.shape[:2]
    crops = []
    dets = []

    if _use_mtcnn and detector is not None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            faces = detector.detect_faces(rgb)
        except Exception as e:
            print("[predict_frame_array] MTCNN detect_faces failed:", e)
            faces = []

        if faces:
            for face in faces:
                box = face.get("box") or face.get("bbox")
                if not box or len(box) < 4:
                    continue
                x, y, wbox, hbox = box
                x, y = int(max(0, x)), int(max(0, y))
                x2 = int(min(w, x + max(1, wbox)))
                y2 = int(min(h, y + max(1, hbox)))
                if x2 <= x or y2 <= y:
                    continue
                crop = rgb[y:y2, x:x2]
                if crop.size == 0:
                    continue
                crops.append(crop)
                dets.append((x, y, x2 - x, y2 - y))
        else:
            crops.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            dets.append(None)
    else:
        crops.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        dets.append(None)

    results = []
    for c in crops:
        try:
            pil = Image.fromarray(c)
            inp = preprocess_pil(pil, img_size)
            pred = model.predict(inp, verbose=0)
            raw, prob_fake, prob_real = interpret_output(pred, invert=invert)
            results.append((float(prob_fake), float(prob_real), float(raw)))
        except Exception as e:
            print("[predict_frame_array] prediction error:", e)
            results.append((0.0, 1.0, 0.0))
    return results, dets

def read_threshold_from_file(thresh_file="best_threshold.txt", default=0.05):
    try:
        if os.path.exists(thresh_file):
            with open(thresh_file, "r") as fh:
                v = float(fh.read().strip())
                print(f"[detect_realtime] Loaded threshold {v} from {thresh_file}")
                return v
    except Exception as e:
        print("[detect_realtime] Could not load threshold file:", e)
    print(f"[detect_realtime] Using default threshold {default}")
    return float(default)

def main(model_path, threshold=None, invert=False, img_size=128, cam_index=0, smooth_frames=25):
    model = load_model_cached(model_path)
    _ = load_calibrator()
    if threshold is None:
        threshold = read_threshold_from_file()
    threshold = float(threshold)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[detect_realtime] Could not open camera.")
        return

    winname = "DeepFake Detector (press q to quit)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    buffer = deque(maxlen=max(1, int(smooth_frames)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            results, dets = predict_frame_array(frame, model, img_size=img_size, invert=invert)

            if results:
                per_face_probs = [p for (p,_,_) in results]
                frame_prob = float(np.mean(per_face_probs)) if len(per_face_probs) > 0 else None
            else:
                frame_prob = None

            if frame_prob is not None:
                buffer.append(frame_prob)

            if len(buffer) > 0:
                agg = float(np.mean(list(buffer)))
                overall_label = "Fake" if agg >= threshold else "Real"
                color = (0,0,255) if overall_label == "Fake" else (0,255,0)
                display_text = f"{overall_label} ({agg:.3f})  threshold={threshold:.3f}"
            else:
                agg = None
                display_text = "No faces"
                color = (200,200,200)

            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            if dets and any(d is not None for d in dets):
                for d in dets:
                    if d is None:
                        continue
                    x,y,wbox,hbox = d
                    cv2.rectangle(frame, (x,y), (x+wbox, y+hbox), (180,180,0), 2)

            cv2.imshow(winname, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[detect_realtime] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="use best_threshold.txt if not provided")
    parser.add_argument("--invert", action="store_true", default=False)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--smooth_frames", type=int, default=25)
    args = parser.parse_args()
    main(args.model, threshold=args.threshold, invert=args.invert, img_size=args.img_size, cam_index=args.cam, smooth_frames=args.smooth_frames)