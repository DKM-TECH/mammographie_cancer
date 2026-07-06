import cv2
import numpy as np


def remove_annotations(img):
    """
    Suppression des annotations type RCC, LCC, RMLO
    via inpainting + détection zones textuelles.
    """

    # 1. Binarisation pour détecter texte
    _, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)

    # 2. Morphologie pour regrouper les caractères
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

    # 3. Inpainting (suppression intelligente)
    cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return cleaned