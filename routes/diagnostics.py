from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


from ml.gradcam import (
    get_img_array,
    make_gradcam_heatmap,
    overlay_heatmap
)

router = APIRouter()

# =========================
# CONFIG
# =========================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]

IMG_SIZE = 224

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "cancer_model.h5"
import os
print("EXISTS ?", os.path.exists(MODEL_PATH))
print("MODEL PATH =", MODEL_PATH)

model = None


# =========================
# LOAD MODEL (lazy)
# =========================

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
    return model


# =========================
# SAVE FILE
# =========================

def save_upload_file(file: UploadFile) -> str:
    ext = file.filename.split(".")[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # =========================
    # NORMALISATION IMAGE
    # =========================
    img = Image.open(file_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(file_path)

    return file_path


# =========================
# PREDICTION SIMPLE
# =========================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Format image non supporté")

    try:
        file_path = save_upload_file(file)

        img_array = get_img_array(file_path)

        model = get_model()
        print("Modèle prêt")

        pred = model.predict(img_array, verbose=0)[0][0]

        score = float(pred)

        if score >= 0.5:
            prediction =  "Cancer [ MALIGNE ]"
            confidence = score
            class_id = 1
        else:
            prediction =  "Cancer [ BELIGNE ]"
            confidence = 1 - score
            class_id = 0

        return {
            "prediction": prediction,
            "confidence": confidence,
            "class_id": class_id,
            "image_path": "/" + file_path.replace("\\", "/")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# PREDICTION + GRAD-CAM
# =========================

@router.post("/predict-gradcam")
async def predict_gradcam(file: UploadFile = File(...)):

    print("===== NOUVELLE REQUETE DIAGNOSTIC =====")
    print("Fichier reçu :", file.filename)
    print("Type :", file.content_type)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Format image non supporté"
        )

    try:

        # =========================
        # SAUVEGARDE IMAGE
        # =========================
        file_path = save_upload_file(file)

        print("Image sauvegardée :", file_path)


        # =========================
        # PREPROCESSING
        # =========================
        img_array = get_img_array(file_path)

        print("Prétraitement terminé")


        # =========================
        # CHARGEMENT MODELE
        # =========================
        model = get_model()

        print("Modèle chargé")


        # =========================
        # PREDICTION
        # =========================
        pred = model.predict(
            img_array,
            verbose=0
        )[0][0]

        score = float(pred)

        print("Score IA :", score)


        # =========================
        # GRAD-CAM
        # =========================
        heatmap = make_gradcam_heatmap(
            img_array,
            model
        )

        print("Heatmap générée")


        gradcam_path = overlay_heatmap(
            file_path,
            heatmap
        )

        print("GradCAM créée :", gradcam_path)


        # =========================
        # INTERPRETATION
        # =========================
        if score >= 0.5:

            prediction = "MALIGNE"
            diagnosis = "Cancer malin détecté"
            confidence = score
            class_id = 1

        else:

            prediction = "BENIGNE"
            diagnosis = "Aucune tumeur maligne détectée"
            confidence = 1 - score
            class_id = 0



        # =========================
        # REPONSE API
        # =========================
        return {

            "status": "success",

            "prediction": prediction,

            "diagnosis": diagnosis,

            "confidence": round(confidence, 4),

            "class_id": class_id,


            "gradcam_image":
                "/" + gradcam_path.replace("\\", "/"),


            "original_image":
                "/" + file_path.replace("\\", "/")

        }


    except Exception as e:

        print("ERREUR DIAGNOSTIC :", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Erreur IA: {str(e)}"
        )