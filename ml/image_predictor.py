import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cancer_model.h5"

print("MODEL PATH =", MODEL_PATH)
print("EXISTS ?", MODEL_PATH.exists())


model = None

IMG_SIZE = 224


# ==========================
# CHARGEMENT MODELE LAZY
# ==========================

def get_model():

    global model

    if model is None:

        import tensorflow as tf

        print("Chargement du modèle EfficientNet...")

        model = tf.keras.models.load_model(
            str(MODEL_PATH)
        )

        print("Modèle chargé")

    return model



# ==========================
# PREPROCESS IMAGE
# ==========================

def preprocess_image(image_file):

    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = tf.keras.utils.load_img(
        image_file,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = tf.keras.utils.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img



# ==========================
# PREDICTION
# ==========================

def predict_image(img_path):

    model = get_model()

    img = preprocess_image(img_path)

    prediction = model.predict(
        img,
        verbose=0
    )

    score = float(prediction[0][0])


    if score >= 0.5:

        prediction_label = "MALIGNE"
        diagnosis = "Cancer malin détecté"
        confidence = score
        class_id = 1

    else:

        prediction_label = "BENIGNE"
        diagnosis = "Aucune tumeur maligne détectée"
        confidence = 1 - score
        class_id = 0


    return {

        "prediction": prediction_label,
        "diagnostic": diagnosis,
        "confidence": confidence,
        "class_id": class_id

    }