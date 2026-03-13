from typing import Union

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) + 1e-12
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def clamp01(x: Union[float, int]) -> float:
    return max(0.0, min(1.0, float(x)))

