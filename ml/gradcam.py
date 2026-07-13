import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.applications.efficientnet import preprocess_input


# =====================================================
# PREPARATION IMAGE
# =====================================================

def get_img_array(img_path, size=(224, 224)):

    print("Chargement image GradCAM...", flush=True)

    img = tf.keras.utils.load_img(
        img_path,
        target_size=size,
        color_mode="rgb"
    )

    img = tf.keras.utils.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    img = img.astype(np.float32)

    print(
        "Image prête :",
        img.shape,
        flush=True
    )

    return img



# =====================================================
# TROUVER EFFICIENTNET
# =====================================================

def get_efficientnet(model):

    """
    Recherche automatique du backbone EfficientNet
    """

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            name = layer.name.lower()

            if "efficientnet" in name:

                return layer


    raise ValueError(
        "Backbone EfficientNet introuvable"
    )



# =====================================================
# TROUVER DERNIERE COUCHE CONVOLUTIONNELLE
# =====================================================

def get_last_conv_layer(base_model):

    """
    EfficientNetB0 :
    dernière couche convolutionnelle = top_conv
    """

    try:

        return base_model.get_layer(
            "top_conv"
        )

    except:

        pass


    # recherche automatique si nom différent

    for layer in reversed(base_model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            return layer


    raise ValueError(
        "Aucune couche convolutionnelle trouvée"
    )



# =====================================================
# GRAD-CAM
# =====================================================

def make_gradcam_heatmap(img_array, model):

    print("=== GRADCAM START ===", flush=True)


    # Trouver EfficientNet
    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "efficientnet" in layer.name.lower():
                base_model = layer
                break


    if base_model is None:
        raise ValueError(
            "EfficientNet introuvable"
        )


    print(
        "EfficientNet :",
        base_model.name,
        flush=True
    )


    # Dernière couche convolutionnelle
    last_conv = base_model.get_layer(
        "top_conv"
    )


    print(
        "Dernière conv :",
        last_conv.name,
        flush=True
    )


    # =================================================
    # IMPORTANT :
    # On reconstruit le chemin complet
    # =================================================

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv.output,
            base_model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_output, features = grad_model(
            img_array,
            training=False
        )


        # passer dans la tête de classification
        x = features


        start = False

        for layer in model.layers:

            if layer == base_model:
                start = True
                continue


            if start:
                x = layer(x)


        loss = x[:,0]



    grads = tape.gradient(
        loss,
        conv_output
    )


    if grads is None:

        raise Exception(
            "Gradient nul"
        )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_output = conv_output[0]


    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )


    print(
        "=== HEATMAP OK ===",
        flush=True
    )


    return heatmap.numpy()

# =====================================================
# SUPERPOSITION IMAGE + HEATMAP
# =====================================================

def overlay_heatmap(
        img_path,
        heatmap,
        alpha=0.45
):


    img = cv2.imread(
        str(img_path)
    )


    if img is None:

        raise ValueError(
            f"Image impossible à lire : {img_path}"
        )



    h, w = img.shape[:2]


    heatmap = cv2.resize(
        heatmap,
        (w, h)
    )


    heatmap = np.uint8(
        255 *
        heatmap
    )


    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    gradcam = cv2.addWeighted(
        img,
        1-alpha,
        heatmap_color,
        alpha,
        0
    )



    original = Path(
        img_path
    )


    output = original.parent / (
        original.stem +
        "_gradcam" +
        original.suffix
    )


    cv2.imwrite(
        str(output),
        gradcam
    )


    print(
        "GradCAM sauvegardée :",
        output,
        flush=True
    )


    return str(output)