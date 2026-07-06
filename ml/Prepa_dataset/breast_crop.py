import cv2
import numpy as np


def crop_breast(img):
    """
    Détection du sein via contours principaux.
    """

    # binarisation inversée (sein = zone claire)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # fermeture morphologique
    kernel = np.ones((15, 15), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # fallback

    # plus grand contour = sein
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    cropped = img[y:y+h, x:x+w]

    return cropped