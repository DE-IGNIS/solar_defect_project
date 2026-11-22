
import cv2
import numpy as np

IMG_H = 64
IMG_W = 96

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(4, 4)
)


def preprocess_thermal(gray, use_clahe=True):
    """
    Input:
        grayscale uint8 image (H, W)

    Output:
        normalized float32 image (3, 64, 96)
    """

    if gray is None:
        raise ValueError("Could not load thermal image")

    if use_clahe:
        gray = _clahe.apply(gray)

    gray = cv2.resize(
        gray,
        (IMG_W, IMG_H),
        interpolation=cv2.INTER_CUBIC
    )

    # Convert grayscale into 3 channels
    x = np.stack(
        [gray.astype(np.float32) / 255.0] * 3,
        axis=0
    )

    mean = np.array(MEAN, np.float32)[:, None, None]
    std = np.array(STD, np.float32)[:, None, None]

    return (x - mean) / std
