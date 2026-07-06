import cv2
import numpy as np

def crop_and_center(img):

    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # plus grande zone = sein
    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)

    roi = img[y:y+h, x:x+w]

    # centrer dans une image carrée
    size = max(w, h)

    canvas = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi

    return canvas

def remove_black_background(img):

    _, mask = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_and(img, img, mask=mask)

    return result