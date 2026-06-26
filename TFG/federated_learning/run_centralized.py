# -------------------------------------------------------------------------------------------------------------
# File Name                : run_centralized.py
# Author                   : Clara Benejam Pons
# Description              : Centralized baseline — all 5 flocks merged into a single dataset.
#                            Split: 70% train / 15% val / 15% test (stratified by label).
#                            Produces metrics + comparison plots equivalent to the federated run.
# Usage                    : python federated_learning/run_centralized.py
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from data_loader import load_federated_clients
from model.clasification_model import build_model, NUM_CLASSES, BATCH_SIZE

# ============================================================
# CONFIG
# ============================================================
CLEAN_DATA_PATH = "data2"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15         

EPOCHS      = 50
PATIENCE    = 10           
RANDOM_SEED = 42

RESULTS_DIR = "results/centralized"

# ============================================================
# HELPERS
# ============================================================

def merge_clients(clients_data):
    """
    Function: Concatenate X and y arrays from all clients into one dataset.
    Args: clients_data (dict): Dictionary of client dataframes.
    Returns: (X_all, y_all)
    """
    X_all, y_all = [], []
    for client_id, df in clients_data.items():
        X_all.append(df["X"].values if "X" in df else df.values[:, :-1])
        y_all.append(df["y"].values if "y" in df else df.values[:, -1])
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def load_and_merge(data_path):
    """
    Function: Load all clients' data and merge into a single dataset.
    Args: data_path (str): Path to the directory containing client data.
    Return: (X_all, y_all)
    """
    from data_loader import prepare_client_splits, prepare_client_tensors

    clients_data = load_federated_clients(data_path)
    print(f"\n  Clientes encontrados: {list(clients_data.keys())}")

    client_splits  = prepare_client_splits(clients_data, random_seed=RANDOM_SEED)
    client_tensors = prepare_client_tensors(client_splits)

    # Merge all splits back into raw arrays — we will re-split centrally
    X_parts, y_parts = [], []
    for cid, data in client_tensors.items():
        for split in ("X_train", "X_val", "X_test"):
            X_parts.append(data[split])
        for split in ("y_train", "y_val", "y_test"):
            y_parts.append(data[split])
        print(f"    [{cid}] "
              f"train={len(data['X_train'])}  "
              f"val={len(data['X_val'])}  "
              f"test={len(data['X_test'])}")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0).astype(int)

    return X_all, y_all


def stratified_split(X, y, train_ratio, val_ratio, random_seed):
    """
    Function: Stratified 3-way split: train / val / test.
    Args: X (np.ndarray): Features.
          y (np.ndarray): Labels.
          train_ratio (float): Proportion of data for training.
          val_ratio (float): Proportion of data for validation.
          random_seed (int): Random seed for reproducibility.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    # First cut: train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=random_seed)

    # Second cut: val vs test (proportional inside the remainder)
    val_of_tmp = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_of_tmp),
        stratify=y_tmp,
        random_state=random_seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def add_bar_labels(ax, bars):
    """Write numeric value (5 decimals) above each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.5f}",
            ha="center", va="bottom",
            fontsize=6.5, rotation=90,
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n" + "=" * 70)
    print("CENTRALIZED TRAINING — ALL 5 FLOCKS MERGED")
    print("=" * 70)
    print(f"  Split     : {int(TRAIN_RATIO*100)}% train / "
          f"{int(VAL_RATIO*100)}% val / "
          f"{int(TEST_RATIO*100)}% test  (stratified)")
    print(f"  Epochs    : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Data path : {CLEAN_DATA_PATH}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & merge
    # ------------------------------------------------------------------
    print("\n1. Cargando y fusionando datos...")
    X_all, y_all = load_and_merge(CLEAN_DATA_PATH)
    print(f"\n  Total muestras: {len(X_all)}  |  Shape: {X_all.shape}")
    unique, counts = np.unique(y_all, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Clase {cls}: {cnt} ({cnt/len(y_all)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Stratified split
    # ------------------------------------------------------------------
    print("\n2. Dividiendo dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X_all, y_all, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )
    print(f"  Train : {len(X_train):>6} muestras  ({len(X_train)/len(X_all)*100:.1f}%)")
    print(f"  Val   : {len(X_val):>6} muestras  ({len(X_val)/len(X_all)*100:.1f}%)")
    print(f"  Test  : {len(X_test):>6} muestras  ({len(X_test)/len(X_all)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Build model & prepare data
    # ------------------------------------------------------------------
    print("\n3. Construyendo modelo...")
    input_shape = X_train.shape[1:]
    model = build_model(input_shape=input_shape, num_classes=NUM_CLASSES)
    model.summary()

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)
    y_test_cat  = to_categorical(y_test,  num_classes=NUM_CLASSES)

    class_weights = get_class_weights(y_train)
    print(f"\n  Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # 4. Train with early stopping (save best by val_loss)
    # ------------------------------------------------------------------
    print("\n4. Entrenando modelo centralizado...")

    best_weights_path = os.path.join(RESULTS_DIR, "best_centralized_weights.h5")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Best epoch = epoch with lowest val_loss
    best_epoch = int(np.argmin(history.history["val_loss"]))
    print(f"\n  Mejor época (val_loss mínimo): {best_epoch + 1}")

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    print("\n5. Evaluando en test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Test accuracy   : {test_acc:.5f}")
    print(f"  Test loss       : {test_loss:.5f}")
    print(f"  Macro F1        : {macro_f1:.5f}")
    print(f"  Weighted F1     : {weighted_f1:.5f}")
    print("\n  Classification report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # 6. Save metrics
    # ------------------------------------------------------------------
    print("\n6. Guardando métricas...")
    history_df = pd.DataFrame(history.history)
    history_df.index.name = "epoch"
    history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"))

    # Summary table (best epoch)
    summary = {
        "split_train":      f"{int(TRAIN_RATIO*100)}%",
        "split_val":        f"{int(VAL_RATIO*100)}%",
        "split_test":       f"{int(TEST_RATIO*100)}%",
        "total_samples":    len(X_all),
        "train_samples":    len(X_train),
        "val_samples":      len(X_val),
        "test_samples":     len(X_test),
        "epochs_trained":   len(history.history["loss"]),
        "best_epoch":       best_epoch + 1,
        "best_train_accuracy":  float(history.history["accuracy"][best_epoch]),
        "best_train_loss":      float(history.history["loss"][best_epoch]),
        "best_val_accuracy":    float(history.history["val_accuracy"][best_epoch]),
        "best_val_loss":        float(history.history["val_loss"][best_epoch]),
        "test_accuracy":    float(test_acc),
        "test_loss":        float(test_loss),
        "Macro_F1":         float(macro_f1),
        "Weighted_F1":      float(weighted_f1),
        "Epocas_locales":   EPOCHS,
        "Mean_accuracy":    float(test_acc),
        "Mean_Macro_F1":    float(macro_f1),
        "Mean_Weighted_F1": float(weighted_f1),
    }

    summary_path = os.path.join(RESULTS_DIR, "summary_centralized.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    with open(os.path.join(RESULTS_DIR, "summary_centralized.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f" Tabla resumen guardada en: {summary_path}")

    