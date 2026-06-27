# -------------------------------------------------------------------------------------------------------------
# File Name                : centralizado.py
# Author                   : Clara Benejam Pons
# Description              : centralized training of a single model for all herds in the dataset.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.clasification_model import build_model, BATCH_SIZE

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "data_muresk"   
RESULTS_DIR = "results/muresk_single_model"

EPOCHS = 50
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# ============================================================
# LOAD FILES
# ============================================================

def read_file(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)

    if path.endswith(".csv"):
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path)

    raise ValueError(f"Formato no soportado: {path}")


def load_all_sheep_files(data_root):
    files = []

    for ext in ["*.xlsx", "*.xls", "*.csv"]:
        files.extend(glob.glob(os.path.join(data_root, "*", ext)))

    if len(files) == 0:
        raise ValueError(f"No se encontraron Excel/CSV dentro de {data_root}/*/")

    dfs = []

    for path in files:
        df = read_file(path)

        herd_name = os.path.basename(os.path.dirname(path))
        sheep_file = os.path.basename(path)

        df["herd_id"] = herd_name
        df["sheep_file"] = sheep_file

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    print(f"Archivos cargados: {len(files)}")
    print(f"Filas totales: {len(full_df)}")
    print("Columnas:")
    print(full_df.columns.tolist())

    return full_df


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_windowed_dataset(df):
    x_cols = [f"x_{i}" for i in range(300)]
    y_cols = [f"y_{i}" for i in range(300)]
    z_cols = [f"z_{i}" for i in range(300)]

    missing = [c for c in x_cols + y_cols + z_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de señal. Ejemplo: {missing[:10]}")

    X_x = df[x_cols].values
    X_y = df[y_cols].values
    X_z = df[z_cols].values

    X = np.stack([X_x, X_y, X_z], axis=2)
    X = X.astype(np.float32)

    X = X[..., np.newaxis]

    if "label" in df.columns:
        labels = df["label"].values
    elif "label_name" in df.columns:
        labels = df["label_name"].values
    else:
        label_candidates = ["sitting", "standing", "walking", "grazing", "ruminating"]
        if all(c in df.columns for c in label_candidates):
            labels = df[label_candidates].idxmax(axis=1).values
        else:
            raise ValueError("No encuentro columna 'label', 'label_name' ni columnas one-hot de actividad.")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    return X, y, le


def normalize_X(X_train, X_val, X_test):
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(n_val, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)

    X_train = X_train_flat.reshape(X_train.shape).astype(np.float32)
    X_val = X_val_flat.reshape(X_val.shape).astype(np.float32)
    X_test = X_test_flat.reshape(X_test.shape).astype(np.float32)

    return X_train, X_val, X_test


def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n1. Cargando todos los Excel/CSV")
    df = load_all_sheep_files(DATA_ROOT)

    print("\n2. Preparando tensores")
    X, y, le = prepare_windowed_dataset(df)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    print("\nClases:")
    for idx, cls in enumerate(le.classes_):
        print(f"  {idx}: {cls}")

    print("\nDistribución de clases:")
    print(pd.Series(y).value_counts(normalize=True).sort_index())

    num_classes = len(le.classes_)

    print("\n3. Split train / val / test")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape} | y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape} | y_test : {y_test.shape}")

    print("\n4. Normalizando")
    X_train, X_val, X_test = normalize_X(X_train, X_val, X_test)

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    class_weight = get_class_weights(y_train)

    print("\nClass weights:")
    print(class_weight)

    print("\n5. Entrenando modelo único centralizado")
    tf.keras.backend.clear_session()

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    print("\n6. Evaluando")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 70)
    print("RESULTADOS MODELO ÚNICO CENTRALIZADO ")
    print("=" * 70)
    print(f"Test loss       : {test_loss:.4f}")
    print(f"Mean accuracy   : {accuracy:.4f}")
    print(f"Mean Macro F1   : {macro_f1:.4f}")
    print(f"Mean Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame([{
        "algorithm": "Single centralized model",
        "aggregation": "None",
        "mu": "-",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "test_loss": test_loss,
        "epochs_trained": len(history.history["loss"]),
        "num_classes": num_classes,
        "classes": ", ".join([str(c) for c in le.classes_])
    }])

    results_path = os.path.join(RESULTS_DIR, "muresk_single_model_results.csv")
    model_path = os.path.join(RESULTS_DIR, "muresk_single_model.keras")

    results_df.to_csv(results_path, index=False)
    model.save(model_path)

    print("\nGuardado:")
    print(f"  Resultados: {results_path}")
    print(f"  Modelo    : {model_path}")