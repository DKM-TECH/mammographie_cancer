import cv2
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


def get_img_array(img_path, size=(224,224)):

    print("Chargement image...", flush=True)

    img = tf.keras.utils.load_img(
        img_path,
        target_size=size,
        color_mode="rgb"
    )

    img = tf.keras.utils.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    img = img.astype(np.float32)

    print("Image prête :", img.shape, flush=True)

    return img

# ========================================================
# TROUVER DERNIÈRE COUCHE CONV (ROBUSTE)
# ========================================================

def get_last_conv_layer(model):
    """
    Fonction robuste pour modèles imbriqués (EfficientNet dans Sequential)
    """
    last_conv = None

    for layer in model.layers:

        # CAS 1 : modèle imbriqué (Sequential / EfficientNet)
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if len(getattr(sublayer, "output_shape", [])) == 4:
                    last_conv = sublayer

        # CAS 2 : layer direct
        if len(getattr(layer, "output_shape", [])) == 4:
            last_conv = layer

    if last_conv is None:
        raise ValueError("Aucune couche convolutionnelle trouvée")

    return last_conv


# ========================================================
# GRAD-CAM CORE
# ========================================================
def make_gradcam_heatmap(img_array, model):

    import tensorflow as tf
    import numpy as np


    # ================================
    # Trouver EfficientNet
    # ================================

    base_model = model.get_layer("efficientnetb0")


    # Dernière couche convolutionnelle
    last_conv_layer = base_model.get_layer(
        "top_conv"
    )


    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )


    # ================================
    # Gradient
    # ================================

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )


        loss = predictions[:,0]


    grads = tape.gradient(
        loss,
        conv_outputs
    )


    if grads is None:
        raise Exception(
            "Gradients NULL : impossible de calculer GradCAM"
        )


    # ================================
    # Calcul heatmap
    # ================================

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]


    heatmap = tf.squeeze(
        heatmap
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )


    return heatmap.numpy()
# ========================================================
# SUPERPOSITION HEATMAP
# ========================================================




def overlay_heatmap(img_path, heatmap, alpha=0.45):

    # Lecture de l'image originale
    img = cv2.imread(str(img_path))

    if img is None:
        raise ValueError(f"Impossible de lire : {img_path}")

    h, w = img.shape[:2]

    # Redimensionnement de la heatmap à la taille de l'image
    heatmap = cv2.resize(heatmap, (w, h))

    # Normalisation
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)

    # Colorisation
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Fusion image + heatmap
    gradcam = cv2.addWeighted(
        img,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    # Création du chemin de sortie
    original = Path(img_path)

    gradcam_path = original.parent / (
        f"{original.stem}_gradcam{original.suffix}"
    )

    # Sauvegarde
    ok = cv2.imwrite(str(gradcam_path), gradcam)

    if not ok:
        raise RuntimeError(
            f"Impossible d'enregistrer {gradcam_path}"
        )

    print("=" * 60)
    print("IMAGE ORIGINALE :", original)
    print("IMAGE GRADCAM   :", gradcam_path)
    print("FICHIER CREE    :", ok)
    print("EXISTE          :", gradcam_path.exists())
    print("=" * 60)

    return str(gradcam_path)