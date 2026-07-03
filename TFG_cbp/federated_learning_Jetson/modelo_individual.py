# -------------------------------------------------------------------------------------------------------------
# File Name                : modeo_individual.py
# Author                   : Clara Benejam  Pons
# Description              : Train and evaluate an individual model for a specific client using its local data.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from data_loader import (
    load_federated_clients,
    prepare_client_splits,
    prepare_client_tensors,
)

from clasification_model import build_model, NUM_CLASSES, BATCH_SIZE


CLIENT_ID = "MureskDryPasture"
DATA_PATH = "data/clean"
RESULTS_DIR = "results/jetson"


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\nCargando datos de {CLIENT_ID}...")

clients_data = load_federated_clients(DATA_PATH)
client_splits = prepare_client_splits(clients_data, random_seed=42)
client_tensors = prepare_client_tensors(client_splits)

if CLIENT_ID not in client_tensors:
    raise ValueError(f"No se encontró {CLIENT_ID} en {DATA_PATH}")

data = client_tensors[CLIENT_ID]

X_train = data["X_train"]
y_train = data["y_train"].astype(int)
X_val = data["X_val"]
y_val = data["y_val"].astype(int)
X_test = data["X_test"]
y_test = data["y_test"].astype(int)

y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

class_weight = get_class_weights(y_train)

tf.keras.backend.clear_session()

model = build_model(
    input_shape=X_train.shape[1:],
    num_classes=NUM_CLASSES
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

print("\nEntrenando modelo individual Jetson...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=50,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

results = {
    "client_id": CLIENT_ID,
    "individual_loss": float(test_loss),
    "individual_accuracy": float(acc),
    "individual_macro_f1": float(macro_f1),
    "individual_weighted_f1": float(weighted_f1),
    "best_val_loss": float(np.min(history.history["val_loss"])),
    "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
    "epochs_trained": int(len(history.history["loss"])),
}

csv_path = os.path.join(RESULTS_DIR, "jetson_individual_baseline.csv")
json_path = os.path.join(RESULTS_DIR, "jetson_individual_baseline.json")
model_path = os.path.join(RESULTS_DIR, "jetson_individual_model.keras")

pd.DataFrame([results]).to_csv(csv_path, index=False)

with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

model.save(model_path)

print("\nRESULTADOS JETSON INDIVIDUAL")
print(results)
print(f"\nCSV guardado en: {csv_path}")
print(f"JSON guardado en: {json_path}")
print(f"Modelo guardado en: {model_path}")