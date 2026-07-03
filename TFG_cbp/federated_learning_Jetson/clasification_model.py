# -------------------------------------------------------------------------------------------------------------
# File Name                : clasification_model.py
# Author                   : Clara Benejam Pons
# Description              : Clasificación de 3 clases (activa, pastando, inactiva).
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import glob
import random
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Flatten, Dense, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Reshape, LSTM, Bidirectional
# ============================================================
# CONFIG
# ============================================================

CLEAN_DATA_PATH = "data/clean"
TARGET_STUDY = "Muresk Stubble"
RANDOM_STATE = 42
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 100
TEST_SHEEP_LIST = None
PRINT_PER_FOLD_REPORT = True
SAVE_RESULTS_CSV = True


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
# ============================================================
# LOAD DATA
# ============================================================

def load_study_data(clean_base_path, target_study):
    """
    Function: Load all cleaned CSV files from the specified directory and concatenate them into a single DataFrame.
    Arguments: clean_base_path (str); The base path where the cleaned CSV files are located.
               target_study (str); The name of the study to filter the data by.
    Return: A pandas DataFrame containing the concatenated data from all CSV files for the specified study.
    """
    
    files = glob.glob(os.path.join(clean_base_path, "**", "*.csv"), recursive=True)

    dfs = []
    for file in files:
        df = pd.read_csv(file, sep=";")
        
        if "study_name" not in df.columns:
            print(f"Warning: 'study_name' column not found in {file}. Skipping this file.")
            continue
        
        df = df[df["study_name"] == target_study].copy()
        
        if df.empty:    
            print(f"Warning: No data for study '{target_study}' in {file}. Skipping this file.")
            continue
    
        dfs.append(df)

    if not dfs:
        raise ValueError("No cleaned CSV files were found in data/clean.")

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# X, Y, Z COLUMNS
# ============================================================

def extract_xyz_columns(df):
    """
    Function: Extract and sort the column names for x, y, and z accelerometer data.
    Arguments: df (pandas.DataFrame); The input DataFrame containing the accelerometer data columns.
    Return: Three lists containing the sorted column names for x, y, and z data respectively.
    """
    
    x_cols = sorted(
        [c for c in df.columns if c.startswith("x_")],
        key=lambda c: int(c.split("_")[1])
    )
    y_cols = sorted(
        [c for c in df.columns if c.startswith("y_")],
        key=lambda c: int(c.split("_")[1])
    )
    z_cols = sorted(
        [c for c in df.columns if c.startswith("z_")],
        key=lambda c: int(c.split("_")[1])
    )
    return x_cols, y_cols, z_cols


# ============================================================
# BUILD INPUT TENSOR
# ============================================================

def build_input_tensor(df):
    """"
    Function: Build the input tensor X and label vector y from the DataFrame.
    Arguments: df (pandas.DataFrame); The input DataFrame containing the accelerometer data and labels.
    Return: A tuple (X, y) where X is a numpy array of shape (samples, 300, 4, 1) containing the input data and y is a numpy array of shape (samples,) containing the integer labels.
    """

    df = df[df["label"].notna()].copy()
    
    x_cols, y_cols, z_cols = extract_xyz_columns(df)

    X_x = df[x_cols].to_numpy(dtype=np.float32)
    X_y = df[y_cols].to_numpy(dtype=np.float32)
    X_z = df[z_cols].to_numpy(dtype=np.float32)

    X_magnitude = np.sqrt(X_x**2 + X_y**2 + X_z**2)

    X = np.stack([X_x, X_y, X_z, X_magnitude], axis=2)
    X = np.expand_dims(X, axis=-1)

    y = df["label"].astype(int).to_numpy(dtype=np.int32)

    valid_mask = (y >= 0) & (y < 3)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(y) == 0:
        raise ValueError("No valid labels found in the data after filtering")

    y = y - np.min(y)

    return X, y


# ============================================================
# NORMALIZATION (z-score: medium=0, std=1)
# ============================================================

def normalize_train_val_test(X_train, X_val, X_test):
    """
    Function: Normalize the training, validation, and test datasets using z-score normalization.
    Arguments: X_train (numpy.ndarray); The training data.
               X_val (numpy.ndarray); The validation data.
               X_test (numpy.ndarray); The test data.
    Return: A tuple (X_train, X_val, X_test) containing the normalized data.
    """
    
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


# ============================================================
# MODEL
# ============================================================
def build_model(input_shape=(300, 4, 1), num_classes=3):
    model = Sequential([

        Input(shape=input_shape),

        Conv2D(16, (5,2), activation="relu"),
        Dropout(0.10),

        Conv2D(32, (5,2), activation="relu"),
        Dropout(0.20),

        Conv2D(64, (5,1), activation="relu"),
        Dropout(0.20),
        
        GlobalAveragePooling2D(),

        Dense(64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        ),

        Dropout(0.40),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOSO
# ============================================================

def choose_validation_sheep(all_sheep, test_sheep):
    """
    Function: Choose a validation sheep that is different from the test sheep.
    Arguments: all_sheep (list); A list of all available sheep numbers. 
                test_sheep (int); The sheep number that is being used as the test set.
    Return: An integer representing the sheep number chosen for validation.
    """
    
    if test_sheep not in all_sheep:
        raise ValueError(f"test_sheep {test_sheep} not found in all_sheep.")

    if len(all_sheep) < 2:
        raise ValueError("At least 2 sheep are required.")

    idx = all_sheep.index(test_sheep)

    for offset in range(1, len(all_sheep)):
        candidate = all_sheep[(idx + offset) % len(all_sheep)]
        if candidate != test_sheep:
            return candidate

    raise ValueError("No validation sheep available.")


def evaluate_fold(df_study, test_sheep, val_sheep):
    """
    Function: Evaluate a single LOSO fold by training a model on the training sheep and evaluating on the test sheep.
    Arguments: df_study (pandas.DataFrame); The DataFrame containing the data for the entire study.
               test_sheep (int); The sheep number to be used as the test set.
               val_sheep (int); The sheep number to be used as the validation set.
    Return: A dictionary containing the results of the fold evaluation, including test accuracy, F1 scores, and other relevant metrics.
    """
    
    if test_sheep == val_sheep:
        raise ValueError("test_sheep and val_sheep must be different.")

    df_test = df_study[df_study["sheep_number"] == test_sheep].copy()
    df_val = df_study[df_study["sheep_number"] == val_sheep].copy()
    df_train = df_study[
        (df_study["sheep_number"] != test_sheep) &
        (df_study["sheep_number"] != val_sheep)
    ].copy()

    print(f"\n{'=' * 70}")
    print(f"LOSO fold | test sheep = {test_sheep} | val sheep = {val_sheep}")
    print(f"{'=' * 70}")
    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("One of the train/val/test splits is empty.")

    X_train, y_train = build_input_tensor(df_train)
    X_val, y_val = build_input_tensor(df_val)
    X_test, y_test = build_input_tensor(df_test)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    X_train, X_val, X_test = normalize_train_val_test(X_train, X_val, X_test)

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    class_weight = {
    0: 2.0,   
    1: 1.2,   
    2: 0.8    
    }
    print("Class weights:", class_weight)

    tf.keras.backend.clear_session()

    print("Building model...")
    model = build_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1   
    )
    
    print("Training...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping,  reduce_lr],
        verbose=1
    )

    print("Evaluating on unseen sheep...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Accuracy (sklearn): {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    if PRINT_PER_FOLD_REPORT:
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(
            y_test,
            y_pred,
            target_names=["active", "grazing", "inactive"]
        ))

        print("\n=== CONFUSION MATRIX ===")
        print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    safe_study = TARGET_STUDY.replace(" ", "_")
    model_path = f"models/cnn_{safe_study}_valsheep{val_sheep}_testsheep{test_sheep}.keras"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    fold_result = {
        "test_sheep": int(test_sheep),
        "val_sheep": int(val_sheep),
        "train_samples": int(len(df_train)),
        "val_samples": int(len(df_val)),
        "test_samples": int(len(df_test)),
        "test_loss": float(test_loss),
        "test_accuracy_keras": float(test_acc),
        "test_accuracy_sklearn": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
        "epochs_trained": int(len(history.history["loss"]))
    }

    return fold_result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(RANDOM_STATE)
    print("Loading cleaned data...")
    df = load_study_data(CLEAN_DATA_PATH, TARGET_STUDY)
    print(f"Total samples in study '{TARGET_STUDY}': {len(df)}")

    if df.empty:
        raise ValueError(f"No samples found for study_name = '{TARGET_STUDY}'")

    available_sheep = sorted(df["sheep_number"].dropna().unique().tolist())
    print("Available sheep in this flock:", available_sheep)

    if len(available_sheep) < 3:
        raise ValueError("At least 3 sheep are required for train/val/test splitting.")

    if TEST_SHEEP_LIST is None:
        sheep_to_run = available_sheep
    else:
        sheep_to_run = TEST_SHEEP_LIST
        for sheep_id in sheep_to_run:
            if sheep_id not in available_sheep:
                raise ValueError(f"Sheep {sheep_id} not found in study '{TARGET_STUDY}'")

    print("\nSheep selected for LOSO evaluation:", sheep_to_run)

    all_results = []

    for test_sheep in sheep_to_run:
        val_sheep = choose_validation_sheep(available_sheep, test_sheep)

        try:
            fold_result = evaluate_fold(
                df_study=df,
                test_sheep=test_sheep,
                val_sheep=val_sheep
            )
            all_results.append(fold_result)

        except Exception as e:
            print(f"\nError in fold test_sheep={test_sheep}: {e}")

    if not all_results:
        raise ValueError("No folds completed successfully.")

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FINAL PER-FOLD RESULTS")
    print("=" * 70)
    print(results_df[[
        "test_sheep",
        "val_sheep",
        "test_accuracy_sklearn",
        "macro_f1",
        "weighted_f1",
        "best_val_accuracy",
        "epochs_trained"
    ]])

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Mean accuracy      : {results_df['test_accuracy_sklearn'].mean():.4f} ± {results_df['test_accuracy_sklearn'].std():.4f}")
    print(f"Mean macro F1      : {results_df['macro_f1'].mean():.4f} ± {results_df['macro_f1'].std():.4f}")
    print(f"Mean weighted F1   : {results_df['weighted_f1'].mean():.4f} ± {results_df['weighted_f1'].std():.4f}")
    print(f"Mean best val acc  : {results_df['best_val_accuracy'].mean():.4f} ± {results_df['best_val_accuracy'].std():.4f}")
    print(f"Mean epochs trained: {results_df['epochs_trained'].mean():.2f}")

    if SAVE_RESULTS_CSV:
        os.makedirs("results", exist_ok=True)
        safe_study = TARGET_STUDY.replace(" ", "_")
        csv_path = f"results/loso_results_{safe_study}CM2.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        