import cv2
import numpy as np
import tensorflow as tf
import collections
import os
import mediapipe as mp
from tensorflow.keras.activations import swish

# =========================
# Register EfficientNet's FixedDropout
# =========================
@tf.keras.utils.register_keras_serializable(package="EfficientNet")
class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

# =========================
# Model & Video Settings
# =========================
MODEL_PATH = r"tmp_checkpoint\best_model.keras"  # path to your trained model
VIDEO_PATH = r"test_video.mp4"                   # input video
WRITE_OUT  = True
OUT_PATH   = "outputs/annotated.mp4"
IMG_SIZE   = 128
THRESH     = 0.5
SMOOTH_N   = 12

print("🎯 Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"swish": swish, "FixedDropout": FixedDropout}
)

# =========================
# Mediapipe Face Detection
# =========================
mp_fd = mp.solutions.face_detection
cap = cv2.VideoCapture(VIDEO_PATH)

if WRITE_OUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (W, H))
else:
    writer = None

def preprocess(face_bgr):
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

smooth = collections.deque(maxlen=SMOOTH_N)
with mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        if res.detections:
            for det in res.detections:
                H, W, _ = frame.shape
                b = det.location_data.relative_bounding_box
                x, y, w, h = int(b.xmin * W), int(b.ymin * H), int(b.width * W), int(b.height * H)
                x, y = max(0, x), max(0, y)
                w, h = max(1, w), max(1, h)
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                prob_fake = float(model.predict(preprocess(face), verbose=0)[0][0])
                smooth.append(prob_fake)
                smoothed = np.mean(smooth)

                label = "Fake" if smoothed > THRESH else "Real"
                color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({smoothed:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("DeepFake Detector - Video", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

if writer:
    print("✅ Saved annotated video at:", OUT_PATH)

