"""
==========================================================
ONCO AI - Training Pipeline (EfficientNetB0)
Version propre, corrigée et stable
==========================================================
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ========================================================
# CONFIG
# ========================================================

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

EPOCHS_HEAD = 10
EPOCHS_FINE = 20

TRAIN_DIR = Path(r"D:\TFC & MEMOIRE\UPC L4\YVETTE\app_mamor\Cancer_Train")
TEST_DIR  = Path(r"D:\TFC & MEMOIRE\UPC L4\YVETTE\app_mamor\Cancer_Test")

AUTOTUNE = tf.data.AUTOTUNE

# ========================================================
# DATASET
# ========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ========================================================
# CLASS WEIGHT (CORRIGÉ)
# ========================================================

y_list = []
for _, y in train_ds.unbatch():
    y_list.append(y.numpy())

labels = np.array(y_list)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights_arr))

print("Class weights:", class_weights)

# ========================================================
# PREPROCESSING
# ========================================================

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds  = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ========================================================
# DATA AUGMENTATION
# ========================================================

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ========================================================
# BACKBONE
# ========================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ========================================================
# MODEL
# ========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_aug(inputs)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# ========================================================
# CALLBACKS HEAD
# ========================================================

callbacks_head = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint("best_head.keras", save_best_only=True, monitor="val_auc", mode="max")
]

# ========================================================
# PHASE 1 : HEAD TRAINING
# ========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

print("\n=== PHASE 1 : HEAD TRAINING ===\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks_head
)

# ========================================================
# FINE TUNING
# ========================================================

print("\n=== PHASE 2 : FINE-TUNING ===\n")

base_model.trainable = True

fine_tune_at = int(len(base_model.layers) * 0.9)

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

callbacks_fine = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-7),
    ModelCheckpoint("best_fine.keras", save_best_only=True, monitor="val_auc", mode="max")
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-6),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks_fine
)

# ========================================================
# EVALUATION
# ========================================================

print("\n=== TEST EVALUATION ===\n")

results = model.evaluate(test_ds)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ========================================================
# SAVE MODEL
# ========================================================

save_path = Path(r"D:\TFC_MEMOIRE_UPC_L4\YVETTE\app_mamor\app\ml")

save_path.mkdir(parents=True, exist_ok=True)

file_path = save_path / "cancer_model.keras"

model.save(file_path)

print(f"✔ Modèle sauvegardé ici : {file_path.resolve()}")

print("\nModel saved successfully.")