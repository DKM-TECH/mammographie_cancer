import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input
# =======================================================
# CONFIGURATION PATHS
# =======================================================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR.parent / "Cancer_Train_Preprocessed"
TEST_DIR  = BASE_DIR.parent / "Cancer_Test_Preprocessed"

MODEL_DIR = BASE_DIR.parent / "ml"
MODEL_PATH = MODEL_DIR / "cancer_model.h5"

os.makedirs(MODEL_DIR, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE

# =======================================================
# CLASS WEIGHTS
# =======================================================

from collections import Counter
#import os

print("\nRépartition des images :")

for classe in sorted(os.listdir(TRAIN_DIR)):
    chemin = TRAIN_DIR / classe
    if chemin.is_dir():
        nb = len(list(chemin.glob("*.*")))
        print(f"{classe:20} : {nb}")

# =======================================================
# HEADER
# =======================================================
print("=" * 60)
print("ONCO AI TRAINING")
print("=" * 60)

print(f"BASE_DIR  : {BASE_DIR}")
print(f"TRAIN_DIR : {TRAIN_DIR}")
print(f"TEST_DIR  : {TEST_DIR}")
print(f"MODEL     : {MODEL_PATH}")

# =======================================================
# CHECK DATASET
# =======================================================
assert TRAIN_DIR.exists(), f"TRAIN_DIR introuvable: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"TEST_DIR introuvable: {TEST_DIR}"

print("\n📊 Dataset check:")
print("Train samples:", len(list(TRAIN_DIR.glob("*/*"))))
print("Test samples :", len(list(TEST_DIR.glob("*/*"))))

# =======================================================
# LOAD DATASET
# =======================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16


train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nClasses détectées :", train_ds.class_names)

labels = []

for _, y in train_ds.unbatch():
    labels.append(int(y.numpy()))

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print("\nPoids des classes :")
print(class_weights)

# =======================================================
# PREPROCESSING
# =======================================================
# =======================================================
# PREPROCESSING
# =======================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.20),
    layers.RandomBrightness(0.20),
    layers.RandomTranslation(0.10, 0.10),
])

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.cache()
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.cache()
test_ds = test_ds.prefetch(AUTOTUNE)
#test_ds = test_ds.prefetch(AUTOTUNE)

# =======================================================
# MODEL BUILD
# =======================================================
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # PHASE 1

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
# =======================================================
# METRICS
# =======================================================
metrics = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall")
]

# =======================================================
# CALLBACKS
# =======================================================
callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_auc",
        save_best_only=True,
        mode="max"
    )
]

# =======================================================
# PHASE 1 - TRAIN HEAD ONLY
# =======================================================
print("\n🔵 PHASE 1: training head only")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=metrics
)

history1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights
)

# =======================================================
# PHASE 2 - FINE TUNING
# =======================================================
print("\n🟠 PHASE 2: fine-tuning EfficientNet")

base_model.trainable = True

# freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=metrics
)

history2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    initial_epoch=10,
    callbacks=callbacks,
    class_weight=class_weights
)

# =======================================================
# SAVE FINAL MODE
# # =======================================================
model.save(str(MODEL_PATH))

print("\n✅ TRAINING TERMINÉ")
print(f"📦 Modèle sauvegardé: {MODEL_PATH}")