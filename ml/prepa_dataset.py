"""
=========================================================
ONCO AI
Préparation automatique du dataset
Auteur : Donatien KADIMA
=========================================================
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "Cancer_Train"
TEST_DIR = BASE_DIR / "Cancer_Test"

OUTPUT_TRAIN = BASE_DIR / "Cancer_Train_Cropped"
OUTPUT_TEST = BASE_DIR / "Cancer_Test_Cropped"

IMG_SIZE = 224

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

for folder in [OUTPUT_TRAIN, OUTPUT_TEST]:

    folder.mkdir(exist_ok=True)

    for cls in ["BENIGNE", "MALIGNE"]:
        (folder / cls).mkdir(exist_ok=True)

# =====================================================
# BREAST CROP
# =====================================================

def crop_breast(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image

    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    margin = 20

    x = max(0, x-margin)
    y = max(0, y-margin)

    w = min(image.shape[1]-x, w+2*margin)
    h = min(image.shape[0]-y, h+2*margin)

    crop = image[y:y+h, x:x+w]

    return crop

# =====================================================
# PROCESS DATASET
# =====================================================

def process_folder(input_dir, output_dir):

    print("\nTraitement :", input_dir)

    for cls in ["BENIGNE", "MALIGNE"]:

        files = list((input_dir/cls).glob("*"))

        for file in tqdm(files):

            img = cv2.imread(str(file))

            if img is None:
                continue

            img = crop_breast(img)

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA
            )

            output_path = output_dir / cls / file.name

            cv2.imwrite(str(output_path), img)

# =====================================================
# MAIN
# =====================================================

print("="*60)
print("PREPARATION DU DATASET")
print("="*60)

process_folder(TRAIN_DIR, OUTPUT_TRAIN)

process_folder(TEST_DIR, OUTPUT_TEST)

print("\n======================================")
print("Dataset préparé avec succès !")
print("======================================")

print("\nNouveau dataset :")

print(OUTPUT_TRAIN)

print(OUTPUT_TEST)