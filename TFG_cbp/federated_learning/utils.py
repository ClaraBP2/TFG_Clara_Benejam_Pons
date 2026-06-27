# -------------------------------------------------------------------------------------------------------------
# File Name                : utils.py
# Author                   : Clara Benejam Pons
# Description              : Utility functions for federated learning experiments, including data loading, client splitting, tensor preparation, and model training/evaluation.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es
# -------------------------------------------------------------------------------------------------------------
import os
import sys
import glob
import pandas as pd
import tensorflow as tf

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.clasification_model import (
    extract_xyz_columns,
    build_input_tensor,
    normalize_train_val_test,
    build_model,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS
)

CLEAN_DATA_PATH = "data/clean"
SAVE_RESULTS = True
RESULTS_DIR = "results/federated_learning"
EXPERIMENT_NAME = "fedavg_r30_e1s"

def load_federated_clients(clean_base_path):
    """
    Load all federated clients from the specified base path.
    Each client should have its own folder containing CSV files.
    Returns a dictionary mapping client IDs to their respective DataFrames.
    """
    if not os.path.exists(clean_base_path):
        raise ValueError(f"Path not found: {clean_base_path}")

    clients_data = {}

    for client_folder in sorted(os.listdir(clean_base_path)):
        client_path = os.path.join(clean_base_path, client_folder)

        if not os.path.isdir(client_path):
            continue

        csv_files = glob.glob(os.path.join(client_path, "*.csv"))

        if not csv_files:
            print(f"Warning: no CSV files found in {client_path}")
            continue

        dfs = []
        for file in csv_files:
            print(f"Loading {file}")
            df = pd.read_csv(file, sep=";")
            df["client_id"] = client_folder
            df["source_file"] = os.path.basename(file)
            dfs.append(df)

        client_df = pd.concat(dfs, ignore_index=True)
        clients_data[client_folder] = client_df

    if not clients_data:
        raise ValueError("No client folders with CSV files were found.")

    return clients_data


def summarize_clients(clients_data):
    """
    Function: Print a summary of the loaded federated clients, including the number of samples, unique sheep, study names, and label distribution for each client.
    Args: 
        clients_data: A dictionary mapping client IDs to their respective DataFrames.
    Return: None. Prints the summary to the console.
    """
    print("\n" + "=" * 70)
    print("FEDERATED CLIENT SUMMARY")
    print("=" * 70)

    for client_id, df_client in clients_data.items():
        print(f"\nClient: {client_id}")
        print(f"Samples: {len(df_client)}")

        if "sheep_number" in df_client.columns:
            print(f"Unique sheep: {df_client['sheep_number'].nunique()}")

        if "study_name" in df_client.columns:
            print("Study names:", df_client["study_name"].dropna().unique().tolist())

        if "label" in df_client.columns:
            print("Label distribution:")
            print(df_client["label"].value_counts(normalize=True).sort_index())

def get_client_sheep(df_client):
    """
    Function: Return sorted list of unique sheep in one client.
    Args:
        df_client: DataFrame for a specific client.
    Return:
        List of unique sheep numbers for the client.
    """
    if "sheep_number" not in df_client.columns:
        raise ValueError("'sheep_number' column not found.")

    sheep_list = sorted(df_client["sheep_number"].dropna().unique().tolist())

    if len(sheep_list) < 3:
        raise ValueError("At least 3 sheep are required for train/val/test split.")

    return sheep_list

def choose_validation_sheep(all_sheep, test_sheep):
    """
    Function: Choose one validation sheep different from test sheep.
    Args:
        all_sheep: List of all unique sheep numbers.
        test_sheep: The sheep number to be used for testing.
    Return:
        The sheep number to be used for validation.
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

def split_client_by_sheep(df_client, test_sheep, val_sheep):
    """
    Function: Split a client's DataFrame into train, validation, and test sets based on sheep numbers.
    Args:
        df_client: DataFrame for a specific client.
        test_sheep: The sheep number to be used for testing.
        val_sheep: The sheep number to be used for validation.
    Return:
        Three DataFrames: train_df, val_df, test_df.
    """
    if test_sheep == val_sheep:
        raise ValueError("test_sheep and val_sheep must be different.")

    df_test = df_client[df_client["sheep_number"] == test_sheep].copy()
    df_val = df_client[df_client["sheep_number"] == val_sheep].copy()
    df_train = df_client[
        (df_client["sheep_number"] != test_sheep) &
        (df_client["sheep_number"] != val_sheep)
    ].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("One of the splits is empty.")

    return df_train, df_val, df_test


def prepare_client_splits(clients_data):
    """
    Function: Prepare train/val/test splits for each client based on sheep numbers.
    Args:
        clients_data: A dictionary mapping client IDs to their respective DataFrames.
    Return:
        A dictionary mapping client IDs to their respective train/val/test splits.
    """
    client_splits = {}

    for client_id, df_client in clients_data.items():
        sheep_list = get_client_sheep(df_client)

        test_sheep = sheep_list[0]
        val_sheep = choose_validation_sheep(sheep_list, test_sheep)

        df_train, df_val, df_test = split_client_by_sheep(
            df_client=df_client,
            test_sheep=test_sheep,
            val_sheep=val_sheep
        )

        client_splits[client_id] = {
            "test_sheep": test_sheep,
            "val_sheep": val_sheep,
            "train_df": df_train,
            "val_df": df_val,
            "test_df": df_test
        }

    return client_splits

def summarize_client_splits(client_splits):
    """
    Function: Summarize the train/val/test splits for each client.
    Args:
        client_splits: A dictionary mapping client IDs to their respective train/val/test splits.
    Return:
        None. Prints the summary to the console.
    """
    print("\n" + "=" * 70)
    print("CLIENT TRAIN / VAL / TEST SPLITS")
    print("=" * 70)

    for client_id, split_data in client_splits.items():
        print(f"\nClient: {client_id}")
        print(f"Test sheep: {split_data['test_sheep']}")
        print(f"Val sheep : {split_data['val_sheep']}")
        print(f"Train samples: {len(split_data['train_df'])}")
        print(f"Val samples  : {len(split_data['val_df'])}")
        print(f"Test samples : {len(split_data['test_df'])}")
        
def inspect_labels(clients_data):
    """
    Function: Inspect the labels for each client and print unique label and label_name pairs.
    Args:
        clients_data: A dictionary mapping client IDs to their respective DataFrames.
    Return:
        None. Prints the label inspection to the console.
    """
    print("\n" + "=" * 70)
    print("LABEL INSPECTION")
    print("=" * 70)

    for client_id, df_client in clients_data.items():
        print(f"\nClient: {client_id}")
        if "label" in df_client.columns and "label_name" in df_client.columns:
            print(
                df_client[["label", "label_name"]]
                .drop_duplicates()
                .sort_values("label")
            )
        else:
            print("Columns 'label' or 'label_name' not found.")
            
def prepare_client_tensors(client_splits):
    """
    Function: Convert each client's train/val/test dataframes into normalized tensors.
    Args:
        client_splits: A dictionary mapping client IDs to their respective train/val/test splits.
    Return:
        A dictionary mapping client IDs to their respective tensors.
    """
    client_tensors = {}

    for client_id, split_data in client_splits.items():
        X_train, y_train = build_input_tensor(split_data["train_df"])
        X_val, y_val = build_input_tensor(split_data["val_df"])
        X_test, y_test = build_input_tensor(split_data["test_df"])

        X_train, X_val, X_test = normalize_train_val_test(X_train, X_val, X_test)

        client_tensors[client_id] = {
            "test_sheep": split_data["test_sheep"],
            "val_sheep": split_data["val_sheep"],
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }

    return client_tensors


def summarize_client_tensors(client_tensors):
    """
    Function: Summarize the tensor shapes for each client.
    Args:
        client_tensors: A dictionary mapping client IDs to their respective tensors.
    Return:
        None. Prints the summary to the console.
    """
    print("\n" + "=" * 70)
    print("CLIENT TENSOR SUMMARY")
    print("=" * 70)

    for client_id, tensor_data in client_tensors.items():
        print(f"\nClient: {client_id}")
        print(f"X_train shape: {tensor_data['X_train'].shape}")
        print(f"y_train shape: {tensor_data['y_train'].shape}")
        print(f"X_val shape  : {tensor_data['X_val'].shape}")
        print(f"y_val shape  : {tensor_data['y_val'].shape}")
        print(f"X_test shape : {tensor_data['X_test'].shape}")
        print(f"y_test shape : {tensor_data['y_test'].shape}")
        
def train_one_client(client_id, client_data, local_epochs=5):
    """
    Function: Train and evaluate one local client using its own train/val/test tensors.
    Args:
        client_id: The ID of the client.
        client_data: A dictionary containing the client's train/val/test tensors.
        local_epochs: The number of epochs to train for.
    Return:
        A tuple containing the trained model, training history, and evaluation results.
    """
    X_train = client_data["X_train"]
    y_train = client_data["y_train"]
    X_val = client_data["X_val"]
    y_val = client_data["y_val"]
    X_test = client_data["X_test"]
    y_test = client_data["y_test"]

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=local_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    results = {
        "client_id": client_id,
        "test_sheep": client_data["test_sheep"],
        "val_sheep": client_data["val_sheep"],
        "test_loss": float(test_loss),
        "test_accuracy_keras": float(test_acc),
        "test_accuracy_sklearn": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "epochs_trained": int(len(history.history["loss"]))
    }

    return model, history, results

def run_local_baseline_for_one_client(client_tensors, client_id, local_epochs=5):
    """
    Function: Train and evaluate one local client using its own train/val/test tensors.
    Args:
        client_tensors: A dictionary mapping client IDs to their respective tensors.
        client_id: The ID of the client to train.
        local_epochs: The number of epochs to train for.
    Return:
        A tuple containing the trained model, training history, and evaluation results.
    """
    if client_id not in client_tensors:
        raise ValueError(f"Client '{client_id}' not found.")

    print("\n" + "=" * 70)
    print(f"LOCAL TRAINING FOR CLIENT: {client_id}")
    print("=" * 70)

    model, history, results = train_one_client(
        client_id=client_id,
        client_data=client_tensors[client_id],
        local_epochs=local_epochs
    )

    print("\nRESULTS")
    print(results)

    return model, history, results

def train_all_clients_local(client_tensors, local_epochs=5):
    """
    Function: Train all clients independently and collect their results.
    Args:
        client_tensors: A dictionary mapping client IDs to their respective tensors.
        local_epochs: The number of epochs to train for.
    Return:
        A DataFrame containing the evaluation results for all clients.
    """
    all_results = []

    for client_id in client_tensors.keys():
        print("\n" + "=" * 70)
        print(f"TRAINING LOCAL CLIENT: {client_id}")
        print("=" * 70)

        _, _, results = train_one_client(
            client_id=client_id,
            client_data=client_tensors[client_id],
            local_epochs=local_epochs
        )

        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    return results_df

def summarize_local_results(results_df):
    """
    Function: Summarize the results of local client training.
    Args:
        results_df: A DataFrame containing the evaluation results for all clients.  
    Return:
        None. Prints the summary to the console. 
    """
    print("\n" + "=" * 70)
    print("LOCAL CLIENT RESULTS")
    print("=" * 70)

    print(results_df[[
        "client_id",
        "test_sheep",
        "val_sheep",
        "test_accuracy_sklearn",
        "macro_f1",
        "weighted_f1",
        "epochs_trained"
    ]])

    print("\n" + "=" * 70)
    print("LOCAL SUMMARY")
    print("=" * 70)
    print(f"Mean accuracy    : {results_df['test_accuracy_sklearn'].mean():.4f} ± {results_df['test_accuracy_sklearn'].std():.4f}")
    print(f"Mean macro F1    : {results_df['macro_f1'].mean():.4f} ± {results_df['macro_f1'].std():.4f}")
    print(f"Mean weighted F1 : {results_df['weighted_f1'].mean():.4f} ± {results_df['weighted_f1'].std():.4f}")

def fedavg_aggregate(client_weights, client_sizes):
    """
    Function: Aggregate client weights using the FedAvg algorithm.
    Args:
        client_weights: A list of lists of model weights from each client.
        client_sizes: A list of the number of samples for each client.
    Return:
        A list of aggregated model weights.
    """  
    total_samples = sum(client_sizes)

    aggregated_weights = []
    for weights_list_tuple in zip(*client_weights):
        weighted_layer = np.sum(
            [w * (n / total_samples) for w, n in zip(weights_list_tuple, client_sizes)],
            axis=0
        )
        aggregated_weights.append(weighted_layer)

    return aggregated_weights
def train_client_from_global_weights(client_id, client_data, global_weights, local_epochs=1):
    """
    Function: Train one client starting from the current global model weights.
    Args:
        client_id: The ID of the client.
        client_data: The data for the client.
        global_weights: The current global model weights.
        local_epochs: The number of local epochs to train.
    Return:
        updated_weights: The updated model weights.
        num_samples: The number of samples in the client's dataset.
        metrics: A dictionary containing evaluation metrics.
    """
    X_train = client_data["X_train"]
    y_train = client_data["y_train"]
    X_val = client_data["X_val"]
    y_val = client_data["y_val"]

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(input_shape=X_train.shape[1:], num_classes=NUM_CLASSES)
    model.set_weights(global_weights)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=local_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0
    )

    updated_weights = model.get_weights()
    num_samples = len(X_train)

    last_val_acc = history.history["val_accuracy"][-1]
    last_val_loss = history.history["val_loss"][-1]

    metrics = {
        "client_id": client_id,
        "num_samples": num_samples,
        "val_accuracy": float(last_val_acc),
        "val_loss": float(last_val_loss)
    }

    return updated_weights, num_samples, metrics
def evaluate_global_model_on_clients(global_model, client_tensors):
    """
    Function: Evaluate the global model on each client's test set and collect metrics.
    Args:
        global_model: The global model to evaluate.
        client_tensors: A dictionary mapping client IDs to their respective tensors.
    Return:
        A DataFrame containing the evaluation results for all clients.
    """
    all_results = []

    for client_id, client_data in client_tensors.items():
        X_test = client_data["X_test"]
        y_test = client_data["y_test"]
        y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

        test_loss, test_acc = global_model.evaluate(X_test, y_test_cat, verbose=0)

        y_pred_probs = global_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        all_results.append({
            "client_id": client_id,
            "test_loss": float(test_loss),
            "test_accuracy_keras": float(test_acc),
            "test_accuracy_sklearn": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        })

    return pd.DataFrame(all_results)

def run_fedavg_simulation(client_tensors, num_rounds=3, local_epochs=1, mu=0.0001):
    """
    Function: Run a simple FedAvg simulation.
    Args:
        client_tensors: A dictionary mapping client IDs to their respective tensors.
        num_rounds: The number of federated rounds to perform.
        local_epochs: The number of local epochs to train each client.
        mu: The regularization parameter for the FedProx algorithm.
    Return:
        global_model: The final global model.
        history_df: A DataFrame containing the training history.
        final_results_df: A DataFrame containing the final evaluation results.
    """
    first_client_id = list(client_tensors.keys())[0]
    input_shape = client_tensors[first_client_id]["X_train"].shape[1:]

    global_model = build_model(input_shape=input_shape, num_classes=NUM_CLASSES)
    global_weights = global_model.get_weights()

    round_history = []

    for rnd in range(1, num_rounds + 1):
        print("\n" + "=" * 70)
        print(f"FEDERATED ROUND {rnd}/{num_rounds}")
        print("=" * 70)

        client_weights = []
        client_sizes = []
        round_metrics = []

        for client_id, client_data in client_tensors.items():
            updated_weights, num_samples, metrics = train_client_from_global_weights(
                client_id=client_id,
                client_data=client_data,
                global_weights=global_weights,
                local_epochs=local_epochs,
                mu = mu
            )

            client_weights.append(updated_weights)
            client_sizes.append(num_samples)
            round_metrics.append(metrics)

            print(
                f"Client: {client_id} | "
                f"samples: {num_samples} | "
                f"val_acc: {metrics['val_accuracy']:.4f} | "
                f"val_loss: {metrics['val_loss']:.4f}"
            )

        global_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.set_weights(global_weights)

        round_df = evaluate_global_model_on_clients(global_model, client_tensors)
        mean_acc = round_df["test_accuracy_sklearn"].mean()
        mean_macro_f1 = round_df["macro_f1"].mean()

        print(f"\nRound {rnd} mean test accuracy: {mean_acc:.4f}")
        print(f"Round {rnd} mean macro F1    : {mean_macro_f1:.4f}")

        round_history.append({
            "round": rnd,
            "mean_test_accuracy": float(mean_acc),
            "mean_macro_f1": float(mean_macro_f1)
        })

    history_df = pd.DataFrame(round_history)
    final_results_df = evaluate_global_model_on_clients(global_model, client_tensors)

    return global_model, history_df, final_results_df

def summarize_federated_results(history_df, final_results_df):
    """
    Function: Summarize the results of the federated learning simulation.
    Args:
        history_df: A DataFrame containing the training history of each federated round.
        final_results_df: A DataFrame containing the final evaluation results for all clients.
    Return:
        None. Prints the summary to the console.
    """
    print("\n" + "=" * 70)
    print("FEDERATED ROUND HISTORY")
    print("=" * 70)
    print(history_df)

    print("\n" + "=" * 70)
    print("FINAL GLOBAL MODEL RESULTS BY CLIENT")
    print("=" * 70)
    print(final_results_df)

    print("\n" + "=" * 70)
    print("FINAL FEDERATED SUMMARY")
    print("=" * 70)
    print(f"Mean accuracy    : {final_results_df['test_accuracy_sklearn'].mean():.4f} ± {final_results_df['test_accuracy_sklearn'].std():.4f}")
    print(f"Mean macro F1    : {final_results_df['macro_f1'].mean():.4f} ± {final_results_df['macro_f1'].std():.4f}")
    print(f"Mean weighted F1 : {final_results_df['weighted_f1'].mean():.4f} ± {final_results_df['weighted_f1'].std():.4f}")
     
def ensure_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
def save_federated_results(history_df, final_results_df, experiment_name, results_dir=RESULTS_DIR):
    """
    Function: Save the federated learning results to CSV and TXT files.
    Args:   
        history_df: A DataFrame containing the training history of each federated round.
        final_results_df: A DataFrame containing the final evaluation results for all clients.
        experiment_name: The name of the experiment (used for file naming).
        results_dir: The directory where results will be saved.
    Return:
        None. Saves the results to files in the specified directory.
    """
    ensure_results_dir(results_dir)

    history_path = os.path.join(results_dir, f"{experiment_name}_2historyblstm.csv")
    final_path = os.path.join(results_dir, f"{experiment_name}_2final_resultsblstm.csv")
    summary_path = os.path.join(results_dir, f"{experiment_name}_2summaryblstm.txt")

    history_df.to_csv(history_path, index=False)
    final_results_df.to_csv(final_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FEDERATED SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Mean accuracy    : {final_results_df['test_accuracy_sklearn'].mean():.4f} ± {final_results_df['test_accuracy_sklearn'].std():.4f}\n")
        f.write(f"Mean macro F1    : {final_results_df['macro_f1'].mean():.4f} ± {final_results_df['macro_f1'].std():.4f}\n")
        f.write(f"Mean weighted F1 : {final_results_df['weighted_f1'].mean():.4f} ± {final_results_df['weighted_f1'].std():.4f}\n")

    print(f"\nSaved history to: {history_path}blstm")
    print(f"Saved final results to: {final_path}blstm")
    print(f"Saved summary to: {summary_path}blstm")


if __name__ == "__main__":
    clients_data = load_federated_clients(CLEAN_DATA_PATH)
    summarize_clients(clients_data)
    inspect_labels(clients_data)

    client_splits = prepare_client_splits(clients_data)
    summarize_client_splits(client_splits)

    client_tensors = prepare_client_tensors(client_splits)
    summarize_client_tensors(client_tensors)

    global_model, history_df, final_results_df = run_fedavg_simulation(
        client_tensors=client_tensors,
        num_rounds=20,
        local_epochs=1,
        mu=0.0001
    )
    summarize_federated_results(history_df, final_results_df)

    if SAVE_RESULTS:
        save_federated_results(
            history_df=history_df,
            final_results_df=final_results_df,
            experiment_name=EXPERIMENT_NAME
        )