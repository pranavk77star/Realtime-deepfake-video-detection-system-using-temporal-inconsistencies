# diagnostic_cam.py
import cv2, time

backends = [
    ("CAP_ANY", cv2.CAP_ANY),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_FFMPEG", cv2.CAP_FFMPEG),
]

print("Testing camera indexes 0..4 with backends:", [b[0] for b in backends])
for idx in range(5):
    for name, backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
            opened = cap.isOpened()
            read_ok = False
            if opened:
                ret, _ = cap.read()
                read_ok = bool(ret)
            print(f"index={idx} backend={name:8s} opened={opened} read_ok={read_ok}")
            if opened:
                cap.release()
        except Exception as e:
            print(f"index={idx} backend={name:8s} ERROR {e}")
        time.sleep(0.05)
print("DONE")
