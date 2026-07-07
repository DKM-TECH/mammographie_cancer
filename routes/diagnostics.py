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
    try:
        print("1. Début", flush=True)

        filepath = save_upload_file(file)
        print("2. Image sauvegardée :", filepath, flush=True)

        img_array = get_img_array(filepath)
        print("3. Prétraitement OK", flush=True)

        model = get_model()
        print("4. Modèle chargé", flush=True)

        pred = float(model.predict(img_array, verbose=0)[0][0])
        print("5. Prediction :", pred, flush=True)


        print("6. Début GradCAM", flush=True)

        heatmap = make_gradcam_heatmap(
            img_array,
            model
        )

        print("7. Heatmap créée :", heatmap.shape, flush=True)


        return {
            "status": "ok",
            "prediction": pred,
            "heatmap_shape": str(heatmap.shape)
        }


    except Exception as e:
        import traceback

        print("========== ERREUR COMPLETE ==========", flush=True)
        traceback.print_exc()
        print("=====================================", flush=True)

        return {
        "status": "error",
        "detail": str(e)
        }