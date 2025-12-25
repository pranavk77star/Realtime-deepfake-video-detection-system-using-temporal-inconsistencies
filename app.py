# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sys
from werkzeug.utils import secure_filename
from detect_image import detect_from_path, load_model_cached, load_calibrator

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "fine_tuned_v3.keras")
THRESHOLD = 0.3784
INVERT = True
IMG_SIZE = 128

THRESH_FILE = os.path.join(BASE_DIR, "best_threshold.txt")
if os.path.exists(THRESH_FILE):
    try:
        with open(THRESH_FILE, "r") as fh:
            THRESHOLD = float(fh.read().strip())
            print(f"[app] Using calibrated threshold from {THRESH_FILE}: {THRESHOLD}")
    except Exception:
        print("[app] Could not read best_threshold.txt, using default THRESHOLD")

try:
    _ = load_model_cached(MODEL_PATH)
    _ = load_calibrator()
except Exception as e:
    print("[app] Warning: model/calibrator pre-load failed:", e)

def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    rel = os.path.relpath(MODEL_PATH) if os.path.exists(MODEL_PATH) else MODEL_PATH
    return render_template('about.html', model=rel, threshold=THRESHOLD, invert=INVERT, img_size=IMG_SIZE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if not allowed_file(file.filename):
        return render_template('index.html', error="Unsupported file type. Use jpg/png.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        label, confidence, raw, prob_fake, prob_real = detect_from_path(
            filepath,
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            invert=INVERT,
            img_size=IMG_SIZE
        )
    except Exception as e:
        return render_template('index.html', error=f"Detection error: {e}")

    result = {
        "label": label,
        "prob": float(confidence),
        "raw": float(raw),
        "prob_fake": float(prob_fake),
        "prob_real": float(prob_real),
        "threshold": THRESHOLD,
        "invert": INVERT
    }

    image_url = url_for('static', filename=f"uploads/{filename}")
    return render_template('index.html', result=result, image_path=image_url)

import subprocess
_rt_proc = None

@app.route('/realtime', methods=['POST'])
def realtime():
    global _rt_proc
    if _rt_proc and _rt_proc.poll() is None:
        return "Realtime already running."
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "detect_realtime.py"),
        "--model", MODEL_PATH,
        "--threshold", str(THRESHOLD),
        "--img_size", str(IMG_SIZE)
    ]
    if INVERT:
        cmd.append("--invert")
    _rt_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return "Realtime detection started."

@app.route('/stop_realtime', methods=['POST'])
def stop_realtime():
    global _rt_proc
    if _rt_proc and _rt_proc.poll() is None:
        _rt_proc.terminate()
        return "Realtime detection stopped."
    return "No realtime process running."

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)