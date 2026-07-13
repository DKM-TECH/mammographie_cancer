import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from tensorflow.keras.applications.efficientnet import preprocess_input

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR.parent / "Cancer_Test_Preprocessed"

MODEL_PATH = BASE_DIR / "cancer_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ======================================================
# LOAD DATASET
# ======================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

# ======================================================
# LOAD MODEL
# ======================================================

print("Chargement du modèle...")

model = tf.keras.models.load_model(MODEL_PATH)

# ======================================================
# PREDICTIONS
# ======================================================

y_true = []
y_prob = []

start = time.perf_counter()

for images, labels in test_ds:

    probs = model.predict(images, verbose=0)

    y_prob.extend(probs.flatten())

    y_true.extend(labels.numpy())

end = time.perf_counter()

y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)

# ======================================================
# METRICS
# ======================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)

TN, FP, FN, TP = cm.ravel()

specificity = TN / (TN + FP)

# ======================================================
# TEMPS D'INFERENCE
# ======================================================

nb_images = len(y_true)

inference_time = (end - start) / nb_images

# ======================================================
# RESULTS
# ======================================================

print("\n" + "="*60)
print("RESULTATS FINAUX")
print("="*60)

print(f"Accuracy     : {accuracy*100:.2f}%")

print(f"Precision    : {precision*100:.2f}%")

print(f"Recall       : {recall*100:.2f}%")

print(f"Specificity  : {specificity*100:.2f}%")

print(f"F1-score     : {f1:.4f}")

print(f"AUC          : {auc:.4f}")

print(f"Inference    : {inference_time:.4f} seconde/image")

print("\nMatrice de confusion")

print(cm)

print("\nClassification Report")

print(classification_report(
    y_true,
    y_pred,
    target_names=["BENIGNE", "MALIGNE"]
))