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

from pydantic import BaseModel


class GradcamRequest(BaseModel):
    image_path: str

router = APIRouter()

# =========================
# CONFIG
# =========================


ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]

IMG_SIZE = 224

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "cancer_model.h5"


UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

import os
print("EXISTS ?", os.path.exists(MODEL_PATH))
print("MODEL PATH =", MODEL_PATH)

model = None


# =========================
# LOAD MODEL (lazy)
# =========================

model = None

def get_model():
    global model

    if model is None:
        import tensorflow as tf

        print("Chargement du modèle...", flush=True)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}"
            )

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
        model.summary()
        print("Modèle chargé.", flush=True)

    return model


# =========================
# SAVE FILE
# =========================
def save_upload_file(file: UploadFile):

    extension = Path(file.filename).suffix

    filename = f"{uuid.uuid4().hex}{extension}"

    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(filepath).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img.save(filepath)

    return str(filepath)


# =====================================================
# PREDICTION SIMPLE
# =====================================================

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Format image non supporté"
        )

    try:

        filepath = save_upload_file(file)

        img_array = get_img_array(filepath)

        model = get_model()

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        score = float(prediction)

        if score >= 0.5:
            label = "MALIGNE"
            confidence = score
            class_id = 1
        else:
            label = "BENIGNE"
            confidence = 1 - score
            class_id = 0

        return {

            "status": "success",

            "prediction": label,

            "confidence": round(confidence,4),

            "class_id": class_id,

            "image_path": "/" + Path(filepath).relative_to(BASE_DIR).as_posix()

        }

    except Exception as e:

        print(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# PREDICTION + GRADCAM
# =====================================================

@router.post("/predict-gradcam")
async def predict_gradcam(file: UploadFile = File(...)):

    filepath = save_upload_file(file)

    img_array = get_img_array(filepath)

    model = get_model()


    score = float(
        model.predict(
            img_array,
            verbose=0
        )[0][0]
    )


    # Réponse immédiate
    return {
        "status":"success",
        "prediction":score,
        "original_image":
            "/" + filepath
    }
@router.post("/generate-gradcam")
async def generate_gradcam(data: GradcamRequest):

    try:

        print("=== GENERATION GRADCAM ===", flush=True)


        # -----------------------------
        # Récupération chemin image
        # -----------------------------

        image_path = data.image_path


        # Si le frontend envoie /uploads/image.png
        # on transforme en chemin serveur réel

        if image_path.startswith("/"):
            image_path = str(BASE_DIR) + image_path


        print(
            "Image utilisée :",
            image_path,
            flush=True
        )


        if not Path(image_path).exists():

            raise FileNotFoundError(
                f"Image introuvable : {image_path}"
            )


        # -----------------------------
        # Chargement modèle
        # -----------------------------

        model = get_model()

        print(
            "Modèle chargé",
            flush=True
        )


        # -----------------------------
        # Prétraitement
        # -----------------------------

        img_array = get_img_array(
            image_path
        )

        print(
            "Image préparée",
            flush=True
        )


        # -----------------------------
        # Génération Heatmap
        # -----------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            model
        )


        print(
            "Heatmap créée",
            flush=True
        )


        # -----------------------------
        # Superposition
        # -----------------------------

        gradcam_path = overlay_heatmap(
            image_path,
            heatmap
        )


        print(
            "GradCAM créée :",
            gradcam_path,
            flush=True
        )


        return {

            "status": "success",

            "gradcam_image":
                "/" + Path(gradcam_path)
                .relative_to(BASE_DIR)
                .as_posix()

        }


    except Exception as e:


        import traceback

        traceback.print_exc()


        raise HTTPException(
            status_code=500,
            detail=str(e)
        )