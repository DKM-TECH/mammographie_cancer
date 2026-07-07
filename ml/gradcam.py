import cv2
import numpy as np
from pathlib import Path

def get_img_array(img_path, size=(224, 224)):

    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = tf.keras.utils.load_img(
        img_path,
        target_size=size,
        color_mode="rgb"
    )

    img = tf.keras.utils.img_to_array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    img = img.astype(np.float32)

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

def make_gradcam_heatmap(img_array, model, pred_index=None):
    import tensorflow as tf
    # Sous-modèle EfficientNet
    efficientnet = model.get_layer("efficientnetb0")

    # Modèle qui renvoie les cartes de caractéristiques
    conv_model = tf.keras.Model(
        inputs=efficientnet.input,
        outputs=efficientnet.get_layer("top_conv").output
    )

    classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])

    x = classifier_input

    # Rejouer la tête de classification
    for layer_name in [
        "global_average_pooling2d",
        "batch_normalization",
        "dropout",
        "dense",
        "dropout_1",
        "dense_1",
    ]:
        x = model.get_layer(layer_name)(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    # Passage avant
    with tf.GradientTape() as tape:

        # La sortie de data_augmentation
        conv_outputs = conv_model(img_array)

        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs)

        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ========================================================
# SUPERPOSITION HEATMAP
# ========================================================

from pathlib import Path


def overlay_heatmap(img_path, heatmap, alpha=0.45):

    img = cv2.imread(str(img_path))

    if img is None:
        raise ValueError("Image introuvable")


    # Redimensionnement image
    img = cv2.resize(
        img,
        (224,224)
    )


    # Redimensionnement heatmap
    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )


    # Conversion en image couleur
    heatmap = np.uint8(
        255 * heatmap
    )


    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # Fusion
    gradcam = cv2.addWeighted(
        img,
        0.55,
        heatmap_color,
        0.45,
        0
    )


    # Création nouveau fichier
    path = Path(img_path)

    output_path = (
        path.parent /
        f"{path.stem}_gradcam{path.suffix}"
    )


    cv2.imwrite(
        str(output_path),
        gradcam
    )


    print("==============================")
    print("IMAGE ORIGINALE :", img_path)
    print("IMAGE GRADCAM   :", output_path)
    print("==============================")


    return str(output_path)