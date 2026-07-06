from Prepa_dataset.utils import read_image, resize, save_image
from Prepa_dataset.remove_labels import remove_annotations
from Prepa_dataset.breast_crop import crop_and_center
from Prepa_dataset.enhance import enhance_image
import os
from tqdm import tqdm

import os
from tqdm import tqdm

valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")



import cv2
import numpy as np


def extract_breast(img):
    """
    Extrait automatiquement la région mammaire.
    Retourne uniquement le sein recadré.
    """

    # Flou léger pour réduire le bruit
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Binarisation
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Recherche des contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img

    # Plus grand contour = sein
    largest = max(contours, key=cv2.contourArea)

    # Création du masque
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    # Conserver uniquement le sein
    breast = cv2.bitwise_and(img, img, mask=mask)

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)

    breast = breast[y:y+h, x:x+w]

    return breast


def process_folder(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    print("INPUT DIR:", input_dir)
    print("CONTENT:", os.listdir(input_dir))

    total = 0

    # 1. boucle classes
    for class_name in os.listdir(input_dir):

        class_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"\nClasse trouvée: {class_name}")

        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        files = os.listdir(class_path)

        # 2. boucle images
        for file in tqdm(files, desc=class_name):

            if not file.lower().endswith(valid_ext):
                continue

            in_path = os.path.join(class_path, file)
            out_path = os.path.join(output_class_path, file)

            try:
                img = read_image(in_path)
                img = extract_breast(img)
                #img = remove_annotations(img)
   #             img = crop_and_center(img)
                img = enhance_image(img)
                img = resize(img, (224, 224))

                save_image(out_path, img)

                total += 1

            except Exception as e:
                print("Erreur:", in_path, e)

    print("\nTOTAL IMAGES TRAITÉES:", total)

if __name__ == "__main__":

    process_folder(
        "Cancer_Train",
        "Cancer_Train_Preprocessed"
    )

    process_folder(
        "Cancer_Test",
        "Cancer_Test_Preprocessed"
    )