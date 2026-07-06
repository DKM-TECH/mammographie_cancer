import cv2
import numpy as np


import cv2
import numpy as np


def remove_annotations(img):

    # 1. contraste local (important en mammographie)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    # 2. détection edges (texte = structures fines)
    edges = cv2.Canny(img_eq, 30, 120)

    # 3. dilatation pour couvrir texte
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)

    # 4. inpainting
    cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return cleaned