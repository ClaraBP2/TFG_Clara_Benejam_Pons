#-------------------------------------------------------------------------------------------------------------
# File Name                : client_flower.py
# Author                   : Clara Benejam Pons
# Description              : Data loader for federated learning clients, including train/val/test split and tensor preparation.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import glob

import numpy as np
import pandas as pd

from model.clasification_model import (
    build_input_tensor,
    normalize_train_val_test,
)

import pandas as pd
import json

# =============================================================================
# CLIENT DATA LOADING
# =============================================================================

def load_federated_clients(clean_base_path, client_ids=None):
    """
    Function: Load all federated clients data from client subfolders.
    Args:
        clean_base_path (str): Root directory containing one subfolder per client.
        client_ids (list[str], optional): List of client IDs to load. If None, load all clients.
    Returns:
        dict[str, pd.DataFrame]: Mapping from client_id to its full DataFrame.
    """
    if not os.path.exists(clean_base_path):
        raise ValueError(f"Path not found: {clean_base_path}")

    clients_data = {}

    # Optional filter: load only selected client/rebaño folders.
    # Useful when the Jetson rebaño must be evaluated separately.
    allowed_clients = set(client_ids) if client_ids is not None else None

    for client_folder in sorted(os.listdir(clean_base_path)):
        if allowed_clients is not None and client_folder not in allowed_clients:
            continue
        client_path = os.path.join(clean_base_path, client_folder)

        if not os.path.isdir(client_path):
            continue

        csv_files = glob.glob(os.path.join(client_path, "*.csv"))

        if not csv_files:
            print(f"  Warning: no CSV files found in {client_path}")
            continue

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file, sep=";")
            df = df.copy()
            df["client_id"] = client_folder
            df["source_file"] = os.path.basename(file)
            dfs.append(df)

        clients_data[client_folder] = pd.concat(dfs, ignore_index=True)

    if not clients_data:
        raise ValueError("No client folders with CSV files were found.")

    return clients_data


def summarize_clients(clients_data):
    """
    Function: Print a summary for each client.
    Args:
        clients_data (dict): Output of load_federated_clients.
    """
    print("\n" + "=" * 70)
    print("FEDERATED CLIENT SUMMARY")
    print("=" * 70)

    for client_id, df in clients_data.items():
        print(f"\nClient: {client_id}")
        print(f"  Samples      : {len(df)}")

        if "sheep_number" in df.columns:
            print(f"  Unique sheep : {df['sheep_number'].nunique()}")

        if "study_name" in df.columns:
            print(f"  Studies      : {df['study_name'].dropna().unique().tolist()}")

        if "label" in df.columns:
            print("  Label distribution:")
            print(df["label"].value_counts(normalize=True).sort_index())


def inspect_labels(clients_data):
    """
    Function: Inspect label id / label name consistency across clients.
    Args:
        clients_data (dict): Output of load_federated_clients.
    Return: None
    """
    print("\n" + "=" * 70)
    print("LABEL INSPECTION")
    print("=" * 70)

    for client_id, df in clients_data.items():
        print(f"\nClient: {client_id}")
        if "label" in df.columns and "label_name" in df.columns:
            print(
                df[["label", "label_name"]]
                .drop_duplicates()
                .sort_values("label")
            )
        else:
            print("  Columns 'label' or 'label_name' not found.")


# =============================================================================
# 2. TRAIN / VAL / TEST SPLIT (per sheep)
# =============================================================================

def get_client_sheep(df_client):
    """
    Function: Get sorted list of unique sheep identifiers for a client.
    Args:
        df_client: DataFrame for a single client.
    Return:
        list: List of unique sheep identifiers.
    """
    if "sheep_number" not in df_client.columns:
        raise ValueError("'sheep_number' column not found.")
    sheep_list = sorted(df_client["sheep_number"].dropna().unique().tolist())
    if len(sheep_list) < 3:
        raise ValueError("At least 3 sheep are required for train/val/test split.")
    return sheep_list


def choose_validation_sheep(all_sheep, test_sheep):
    """
    Function: Choose a validation sheep different from the test sheep.
    Args:
        all_sheep (list): List of all sheep identifiers.
        test_sheep: The sheep identifier chosen for testing.
    Return:
        val_sheep: A sheep identifier for validation, different from test_sheep.
    """
    if test_sheep not in all_sheep:
        raise ValueError(f"test_sheep '{test_sheep}' not found in all_sheep.")
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
    Function: Split one client's DataFrame into train/val/test by sheep.
    Args:
        df_client (pd.DataFrame): DataFrame for a single client.
        test_sheep: Sheep identifier for the test set.
        val_sheep: Sheep identifier for the validation set.
    Return:
        tuple: (df_train, df_val, df_test) DataFrames for training, validation, and testing.    
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


def prepare_client_splits(clients_data, random_seed=42):
    """
    Function: Build train/val/test splits for each client using sheep-based splitting.
    Args:
        clients_data (dict): Output of load_federated_clients.
        random_seed (int): Seed for reproducibility.
    Return:
        dict: Mapping from client_id to its train/val/test DataFrames and chosen sheep. 
    """
    rng = np.random.default_rng(random_seed)
    client_splits = {}
    for client_id, df_client in clients_data.items():
        sheep_list = get_client_sheep(df_client)
        
        shuffled = sheep_list.copy()
        rng.shuffle(shuffled)
        test_sheep = shuffled[0]
        val_sheep = choose_validation_sheep(shuffled, test_sheep)

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
    Function: Print train/val/test split summary for each client. 
    Args:
        client_splits (dict): Output of prepare_client_splits.
    Return: None  
    """
    print("\n" + "=" * 70)
    print("CLIENT TRAIN / VAL / TEST SPLITS")
    print("=" * 70)

    for client_id, s in client_splits.items():
        print(f"\nClient: {client_id}")
        print(f"  Test sheep    : {s['test_sheep']}")
        print(f"  Val sheep     : {s['val_sheep']}")
        print(f"  Train samples : {len(s['train_df'])}")
        print(f"  Val samples   : {len(s['val_df'])}")
        print(f"  Test samples  : {len(s['test_df'])}")


# =============================================================================
# 3. TENSOR PREPARATION
# =============================================================================

def prepare_client_tensors(client_splits):
    """
    Function: Convert each client's train/val/test DataFrames into normalized tensors.
    Args:
        client_splits (dict): Output of prepare_client_splits.
    Return:
        dict: Mapping from client_id to its normalized tensors.
    """
    client_tensors = {}

    for client_id, split in client_splits.items():
        X_train, y_train = build_input_tensor(split["train_df"])
        X_val, y_val = build_input_tensor(split["val_df"])
        X_test, y_test = build_input_tensor(split["test_df"])

        X_train, X_val, X_test = normalize_train_val_test(X_train, X_val, X_test)

        client_tensors[client_id] = {
            "test_sheep": split["test_sheep"],
            "val_sheep": split["val_sheep"],
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
    Function: Print tensor shape summary for each client.
    Args:
        client_tensors (dict): Output of prepare_client_tensors.
    Return: None
    """
    print("\n" + "=" * 70)
    print("CLIENT TENSOR SUMMARY")
    print("=" * 70)

    for client_id, t in client_tensors.items():
        print(f"\nClient: {client_id}")
        print(f"  X_train : {t['X_train'].shape}  |  y_train : {t['y_train'].shape}")
        print(f"  X_val   : {t['X_val'].shape}  |  y_val   : {t['y_val'].shape}")
        print(f"  X_test  : {t['X_test'].shape}  |  y_test  : {t['y_test'].shape}")