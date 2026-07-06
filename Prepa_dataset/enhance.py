import cv2
import numpy as np


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def denoise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def enhance_image(img):
    img = apply_clahe(img)
    img = denoise(img)
    return img