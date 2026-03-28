import cv2
import numpy as np

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def stain_normalization(image):
    # Simplified Reinhard normalization
    image = image.astype(np.float32)
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    target_mean = [0.5 * 255] * 3
    target_std = [0.2 * 255] * 3

    normalized = (image - mean) / (std + 1e-8)
    normalized = normalized * target_std + target_mean

    return np.clip(normalized, 0, 255).astype(np.uint8)