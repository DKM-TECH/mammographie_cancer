import cv2
import numpy as np


def read_image(path, grayscale=True):
    """Lecture robuste d'une image médicale"""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise ValueError(f"Image introuvable ou corrompue: {path}")
    return img


def resize(img, size=(224, 224)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize(img):
    """Normalisation [0,1]"""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def save_image(path, img):
    """Sauvegarde sécurisée"""
    cv2.imwrite(path, img)