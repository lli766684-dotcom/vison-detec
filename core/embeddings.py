from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

from utils.math_utils import l2_normalize

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    torch = None
    CLIPModel = None
    CLIPProcessor = None


def legacy_image_embedding(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=32, range=(0.0, 1.0), density=True)
        hist.append(h)
    hist = np.concatenate(hist, axis=0)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    texture = np.array([gx, gy], dtype=np.float32)
    return l2_normalize(np.concatenate([hist.astype(np.float32), texture], axis=0))


def legacy_text_embedding(text: str, dim: int = 98) -> np.ndarray:
    s = text.lower().strip()
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(s.encode("utf-8")):
        vec[i % dim] += float((ch % 31) + 1)
    vec += 0.01
    return l2_normalize(vec)


@dataclass
class VisionBackend:
    name: str
    device: str = "cpu"
    model: Optional[object] = None
    processor: Optional[object] = None

    def __post_init__(self):
        if self.name == "clip":
            if CLIPModel is None or CLIPProcessor is None or torch is None:
                raise RuntimeError("CLIP backend requested but torch/transformers is unavailable")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.model.eval()

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        if self.name == "legacy":
            return legacy_image_embedding(image)
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            vec = feats[0].detach().cpu().numpy().astype(np.float32)
            return l2_normalize(vec)

    def text_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.name == "legacy":
            return np.stack([legacy_text_embedding(t) for t in texts], axis=0)
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            mat = feats.detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            return mat / norms

