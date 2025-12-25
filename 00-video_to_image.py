# 00-video_to_image_args.py
import os
import cv2
import math
import sys

if len(sys.argv) < 3:
    print("Usage: python 00-video_to_image_args.py <input_videos_folder> <output_frames_folder>")
    sys.exit(1)

base_path = sys.argv[1]      # e.g. "C:\\...\\datasets\\Celeb-real"
output_path = sys.argv[2]    # e.g. "C:\\...\\datasets\\frames\\Celeb-real"

print("✅ Script started...")
print("Input folder :", base_path)
print("Output folder:", output_path)

os.makedirs(output_path, exist_ok=True)

def get_filename_only(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Loop through all video files in the folder
video_exts = ('.mp4', '.mov', '.avi', '.mkv')
for filename in sorted(os.listdir(base_path)):
    if not filename.lower().endswith(video_exts):
        continue
    video_file = os.path.join(base_path, filename)
    file_stem = get_filename_only(filename)
    tmp_path = os.path.join(output_path, file_stem)
    # Skip if already processed (basic check)
    if os.path.exists(tmp_path) and len(os.listdir(tmp_path)) > 0:
        print(f"⚠️ Skipping {filename} — output folder exists and not empty.")
        continue

    os.makedirs(tmp_path, exist_ok=True)
    print(f'Processing: {filename}')
    count = 0
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25
    if frame_rate <= 0 or math.isnan(frame_rate):
        frame_rate = 25
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        # Save ~1 frame per second (adjust by changing the divisor)
        if frame_rate > 0 and (frame_id % math.floor(frame_rate) == 0):
            # Defensive: ensure frame valid
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if w < 300:
                scale_ratio = 2
            elif w > 1900:
                scale_ratio = 0.33
            elif 1000 < w <= 1900:
                scale_ratio = 0.5
            else:
                scale_ratio = 1

            width = max(64, int(w * scale_ratio))
            height = max(64, int(h * scale_ratio))
            try:
                new_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print("Resize error:", e)
                continue

            new_filename = f"{file_stem}_{count:03d}.png"
            cv2.imwrite(os.path.join(tmp_path, new_filename), new_frame)
            count += 1
    cap.release()
    print(f"✅ Done {filename}, saved {count} frames")

print("✅ Script finished.")
