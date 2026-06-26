# -------------------------------------------------------------------------------------------------------------
# File Name                : fine_tuning_flower.py
# Author                   : Clara Benejam Pons
# Creation Date            : 2026-03-30
# Description              : Fine-tuning the global model for each client and evaluating its performance.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------
import os
import json
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_loader import (
    load_federated_clients,
    prepare_client_splits,
    prepare_client_tensors,
)

from model.clasification_model import NUM_CLASSES
from evaluation import finetune_global_model

# ============================================================
# PATHS
# ============================================================

CLEAN_DATA_PATH = "data/clean"
MODEL_PATH = "results/flower/best_global_model.keras"
RESULTS_DIR = "results/flower/finetuning"
FINETUNE_EPOCHS = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

clients_data = load_federated_clients(CLEAN_DATA_PATH)
client_splits = prepare_client_splits(clients_data, random_seed=42)
client_tensors = prepare_client_tensors(client_splits)

global_model = tf.keras.models.load_model(MODEL_PATH)

results = []

for client_id, data in client_tensors.items():
    print(f"\nFine-tuning local para: {client_id}")

    ft_model = finetune_global_model(
        global_model=global_model,
        client_data=data,
        finetune_epochs=FINETUNE_EPOCHS
    )

    y_test = data["y_test"].astype(int)
    y_pred = np.argmax(ft_model.predict(data["X_test"], verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    results.append({
        "client_id": client_id,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "confusion_matrix": json.dumps(cm)
    })

df = pd.DataFrame(results)
mean_acc = df["test_accuracy"].mean()
mean_macro_f1 = df["test_macro_f1"].mean()
mean_weighted_f1 = df["test_weighted_f1"].mean()

print("\n" + "=" * 60)
print("RESULTADOS FINALES FINE-TUNING")
print("=" * 60)

print(f"Mean Accuracy     : {mean_acc:.4f}")
print(f"Mean Macro F1     : {mean_macro_f1:.4f}")
print(f"Mean Weighted F1  : {mean_weighted_f1:.4f}")

df.to_csv(
    os.path.join(RESULTS_DIR, "finetuning_results.csv"),
    index=False
)

print("\nResultados guardados en results/flower/finetuning/finetuning_results.csv")
print(df)