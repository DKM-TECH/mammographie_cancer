from Prepa_dataset.utils import read_image, resize, save_image
from Prepa_dataset.remove_labels import remove_annotations
from Prepa_dataset.breast_crop import crop_breast
from Prepa_dataset.enhance import enhance_image
import os
from tqdm import tqdm

def process_folder(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir)):

        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        try:
            # 1. load
            img = read_image(in_path)

            # 2. remove annotations
            img = remove_annotations(img)

            # 3. crop breast
            img = crop_breast(img)

            # 4. enhance
            img = enhance_image(img)

            # 5. resize
            img = resize(img, (224, 224))

            # 6. save
            save_image(out_path, img)

        except Exception as e:
            print(f"Erreur sur {file} : {e}")
if __name__ == "__main__":

    process_folder(
        "Cancer_Train",
        "Cancer_Train_Preprocessed"
    )

    process_folder(
        "Cancer_Test",
        "Cancer_Test_Preprocessed"
    )