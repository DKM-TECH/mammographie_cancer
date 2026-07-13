from pathlib import Path
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cancer_model.h5"

model = None

print(">>> model_loader.py chargé <<<", flush=True)
def get_model():

    global model
    print(">>> get_model MODEL_LOADER APPELE <<<", flush=True)
    if model is None:

        print("=" * 60, flush=True)
        print("CHARGEMENT DU MODELE ONCO AI", flush=True)
        print("MODEL :", MODEL_PATH, flush=True)

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
        print("\n===== COUCHES MODELE =====")

        for i, layer in enumerate(model.layers):
             print(
            i,
            layer.name,
            type(layer)
            )

        print("==========================")

        print("MODELE CHARGE", flush=True)

        model.summary(
            expand_nested=True
        )

        print("=" * 60, flush=True)

    return model