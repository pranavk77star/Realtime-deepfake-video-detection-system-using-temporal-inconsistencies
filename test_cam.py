# test_cam.py
import cv2, time, sys

def try_cam(index, backend=None):
    print(f"Trying index={index} backend={backend}")
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        # On Windows use CAP_DSHOW
        cap = cv2.VideoCapture(index, backend)
    ok = cap.isOpened()
    print("isOpened:", ok)
    if not ok:
        return False
    # try read a few frames
    for i in range(5):
        ok, frame = cap.read()
        print(f"read {i}: ok={ok}, frame_none={frame is None}")
        time.sleep(0.2)
    cap.release()
    return True

# try common combos
if __name__ == "__main__":
    # On Windows try cv2.CAP_DSHOW (value 700) or cv2.CAP_MSMF
    backends = [None]
    try:
        backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_MSMF)
    except:
        pass

    for b in backends:
        for idx in range(0,3):
            ok = try_cam(idx, backend=b)
            if ok:
                print("SUCCESS with index", idx, "backend", b)
                sys.exit(0)
    print("No camera found with tested indexes/backends. Check camera/permissions.")
