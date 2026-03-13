from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def preprocess_one(path_str: str, thresholds: Dict) -> Tuple[Image.Image, Dict]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path_str}")

    max_side = int(thresholds["image"]["max_image_side"])
    blur_thr = float(thresholds["quality"]["blur_var_threshold"])
    dark_thr = float(thresholds["quality"]["dark_mean_threshold"])

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)
    gray = arr.mean(axis=2)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    blur_var = float(np.var(dx) + np.var(dy))
    mean_light = float(np.mean(gray))

    flags = {
        "blurry": blur_var < blur_thr,
        "dark": mean_light < dark_thr,
        "blur_var": blur_var,
        "brightness_mean": mean_light,
    }
    return img, flags


def preprocess_images(req: Dict, thresholds: Dict) -> Dict:
    order_reference_processed = {}
    refund_processed = {}
    quality_flags = {}

    for p in req["order_reference_images"]:
        img, flags = preprocess_one(p, thresholds)
        order_reference_processed[p] = img
        quality_flags[p] = flags

    for p in req["refund_images"]:
        img, flags = preprocess_one(p, thresholds)
        refund_processed[p] = img
        quality_flags[p] = flags

    return {
        "order_reference_images_processed": order_reference_processed,
        "refund_images_processed": refund_processed,
        "image_quality_flags": quality_flags,
    }

