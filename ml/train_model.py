import tensorflow as tf
from tensorflow.keras import layers, models
import os

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset/train"
val_dir = "dataset/val"

# =========================
# DATA LOADING
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# normalization
normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# =========================
# TRANSFER LEARNING MODEL
# =========================

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # important pour petit dataset

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAINING
# =========================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("app/ml/cancer_model.h5")

print("Model trained and saved!")