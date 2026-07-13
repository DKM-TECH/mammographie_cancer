import numpy as np
from pathlib import Path
from ml.model_loader import get_model



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