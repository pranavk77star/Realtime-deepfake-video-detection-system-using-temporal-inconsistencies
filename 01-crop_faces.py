# 01-crop_faces.py (final with resume support)
import os
import argparse
import cv2
import numpy as np
from mtcnn import MTCNN
import traceback

def is_valid_image(path):
    try:
        if os.path.getsize(path) == 0:
            return False
        img = cv2.imread(path)
        if img is None:
            return False
        h, w = img.shape[:2]
        return (h > 10 and w > 10)
    except Exception:
        return False

def crop_faces_recursive(input_folder, output_folder, min_conf=0.7, verbose_every=100):
    os.makedirs(output_folder, exist_ok=True)
    detector = MTCNN()
    total_frames = 0
    total_faces = 0
    processed = 0

    for root, _, files in os.walk(input_folder):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue

        rel = os.path.relpath(root, input_folder)
        save_folder = os.path.join(output_folder, rel)
        os.makedirs(save_folder, exist_ok=True)

        for img_file in image_files:
            total_frames += 1
            src = os.path.join(root, img_file)

            # NEW ✅ Skip if already processed
            out_name_check = f"{os.path.splitext(img_file)[0]}_face00.png"
            out_path_check = os.path.join(save_folder, out_name_check)
            if os.path.exists(out_path_check):
                # If at least one face from this frame already exists, skip
                continue

            if not is_valid_image(src):
                print(f"[SKIP] invalid/corrupt image: {src}")
                continue

            try:
                img_bgr = cv2.imread(src)
                if img_bgr is None:
                    print(f"[SKIP] cv2.imread returned None: {src}")
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[ERROR] reading or converting {src}: {e}")
                continue

            # Detect faces
            try:
                results = detector.detect_faces(img)
            except Exception as e:
                print(f"[EXCEPTION] MTCNN failed on {src}: {e}")
                traceback.print_exc()
                continue

            if not results:
                processed += 1
                if processed % verbose_every == 0:
                    print(f"[INFO] processed frames: {processed}, total_faces so far: {total_faces}")
                continue

            face_count_in_frame = 0
            for r in results:
                try:
                    conf = r.get("confidence", 0)
                    if conf < min_conf:
                        continue
                    x, y, w, h = r['box']
                    x, y, w, h = int(x), int(y), int(abs(w)), int(abs(h))
                    mx, my = int(0.3 * w), int(0.3 * h)
                    x1, y1 = max(0, x - mx), max(0, y - my)
                    x2, y2 = min(img.shape[1], x + w + mx), min(img.shape[0], y + h + my)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    out_name = f"{os.path.splitext(img_file)[0]}_face{face_count_in_frame:02d}.png"
                    out_path = os.path.join(save_folder, out_name)
                    cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    face_count_in_frame += 1
                    total_faces += 1
                except Exception as e:
                    print(f"[ERROR] cropping/saving for {src}: {e}")
                    traceback.print_exc()
                    continue

            processed += 1
            if processed % verbose_every == 0:
                print(f"[INFO] processed frames: {processed}, total_faces so far: {total_faces}")

    print(f"\n✅ DONE — frames scanned: {total_frames}, faces saved: {total_faces}")
    return total_frames, total_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Root folder containing dataset frames (walks subfolders)")
    parser.add_argument("output_folder", help="Root folder to save cropped faces (keeps subfolder structure)")
    parser.add_argument("--min_conf", type=float, default=0.7, help="Minimum face detection confidence")
    args = parser.parse_args()

    crop_faces_recursive(args.input_folder, args.output_folder, min_conf=args.min_conf)
