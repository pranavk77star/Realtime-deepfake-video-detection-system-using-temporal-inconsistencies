# tta_detect.py
import numpy as np
from PIL import Image, ImageOps
import argparse
from detect_image import load_model_cached, preprocess_pil, interpret_output, load_calibrator

def tta_predict(img_path, model_path, invert=False, img_size=128, transforms=None):
    model = load_model_cached(model_path)
    calibrator = load_calibrator()
    pil = Image.open(img_path).convert("RGB")
    # define transforms if not provided
    if transforms is None:
        transforms = [
            lambda im: im,
            lambda im: ImageOps.mirror(im),
            lambda im: im.crop((int(im.width*0.05), int(im.height*0.05), int(im.width*0.95), int(im.height*0.95))).resize(im.size),
            lambda im: ImageOps.mirror(im.crop((int(im.width*0.05), int(im.height*0.05), int(im.width*0.95), int(im.height*0.95))).resize(im.size)),
        ]
    probs = []
    raws = []
    for t in transforms:
        try:
            imt = t(pil)
            inp = preprocess_pil(imt, img_size)
            pred = model.predict(inp, verbose=0)
            raw, prob_fake, prob_real = interpret_output(pred, invert=invert)
            p = prob_fake
            # apply calibrator if present
            if calibrator is not None:
                try:
                    if hasattr(calibrator, "predict_proba"):
                        p = float(calibrator.predict_proba([[p]])[:, 1])
                    else:
                        p = float(calibrator.predict([p])[0])
                    p = max(0.0, min(1.0, p))
                except:
                    pass
            probs.append(p)
            raws.append(raw)
        except Exception as e:
            print("transform error:", e)
    if len(probs) == 0:
        raise RuntimeError("No valid TTA predictions")
    avg_p = float(np.mean(probs))
    avg_raw = float(np.mean(raws))
    return avg_raw, avg_p, 1.0 - avg_p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    raw, pf, pr = tta_predict(args.img, args.model, invert=args.invert, img_size=args.img_size)
    label = "Fake" if pf >= 0.0201 else "Real"
    print(f"raw (avg): {raw:.6f}  prob_fake (avg): {pf:.6f}  prob_real: {pr:.6f}  -> {label}")