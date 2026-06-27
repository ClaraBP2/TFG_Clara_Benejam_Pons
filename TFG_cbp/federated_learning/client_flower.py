#-------------------------------------------------------------------------------------------------------------
# File Name                : client_flower.py
# Author                   : Clara Benejam Pons
# Description              : Flower federated client with FedProx local training.
#                            Same file runs on PC (simulated clients) and Jetson Nano (real edge client).
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------

import json
import pandas as pd
import os
import time
import sys
import argparse
import numpy as np
import flwr as fl
import tensorflow as tf
from flwr.common import Scalar
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.clasification_model import build_model, NUM_CLASSES, BATCH_SIZE
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import json

import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
LOCAL_EPOCHS = 4
MU           = 0.05

# =============================================================================
# HELPERS
# =============================================================================
def get_class_weights(y_train):
    """
    Function: estimate class weights for imbalanced datasets.
    Args: 
        y_train: Array of training labels.
    Return: Dictionary mapping class indices to their corresponding weights.
    """"
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# =============================================================================
# FLOWER CLIENT
# =============================================================================

class SheepBehaviorClient(fl.client.NumPyClient):
    """
    Flower client for sheep behavior classification with FedProx local training.
    Works identically on PC (simulated) and Jetson Nano (real edge node).
    """

    def __init__(self, client_id: str, client_data: dict, mu: float = MU):
        self.client_id = client_id
        self.client_data = client_data
        self.mu = mu
        self.input_shape = client_data["X_train"].shape[1:]

        print(f"  [{client_id}] Cliente inicializado | "
              f"train: {len(client_data['X_train'])} | "
              f"val: {len(client_data['X_val'])} | "
              f"test: {len(client_data['X_test'])}")

    def get_parameters(self, config):
        """Return current local model weights."""
        model = build_model(input_shape=self.input_shape, num_classes=NUM_CLASSES)
        return model.get_weights()

    def fit(self, parameters, config):
        """
        Receive global weights, train locally with FedProx, return updated weights.
        """
        X_train = self.client_data["X_train"]
        y_train = self.client_data["y_train"].astype(int)
        X_val   = self.client_data["X_val"]
        y_val   = self.client_data["y_val"].astype(int)

        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val_cat   = to_categorical(y_val,   num_classes=NUM_CLASSES)

        class_weight = get_class_weights(y_train)
        
        MAX_WEIGHT = 2.0

        class_weight = {
            k: min(v, MAX_WEIGHT)
            for k, v in class_weight.items()
        }
        
        print(f"[{self.client_id}] Class weights:", class_weight)

        sample_weights = np.array(
            [class_weight[int(l)] for l in y_train],
            dtype=np.float32
        )

        # Get mu with warmup from config (server can pass it)
        current_mu = float(config.get("mu", self.mu))
        server_round = int(config.get("server_round", 1))
        current_mu = current_mu * min(1.0, server_round / 5)

        # Build model and set global weights
        model = build_model(input_shape=self.input_shape, num_classes=NUM_CLASSES)
        model.set_weights(parameters)

        optimizer = model.optimizer
        loss_fn   = tf.keras.losses.CategoricalCrossentropy()

        # Global weights as tensors for proximal term
        global_weights_tf = [
            tf.convert_to_tensor(w, dtype=tf.float32) for w in parameters
        ]

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train_cat, sample_weights)
        ).batch(BATCH_SIZE)

        best_val_loss = np.inf
        best_weights  = None
        patience = 5
        wait = 0
        history_val_acc  = []
        history_val_loss = []

        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))

        for epoch in range(local_epochs):
            # FedProx training step
            for batch_x, batch_y, batch_sw in train_dataset:
                with tf.GradientTape() as tape:
                    y_pred   = model(batch_x, training=True)
                    ce_loss  = loss_fn(batch_y, y_pred, sample_weight=batch_sw)
                    prox_term = tf.add_n([
                        tf.reduce_mean(tf.square(w_local - w_global))
                        for w_local, w_global in zip(model.trainable_weights, global_weights_tf)
                    ])
                    total_loss = ce_loss + (current_mu / 2.0) * prox_term

                grads = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Validation
            val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
            history_val_loss.append(val_loss)
            history_val_acc.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = model.get_weights()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_weights is None:
            best_weights = model.get_weights()
        model.set_weights(best_weights)

        train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
        val_loss_eval, val_acc_eval = model.evaluate(X_val, y_val_cat, verbose=0)

        y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
        y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

        train_macro_f1 = float(f1_score(y_train, y_train_pred, average="macro", zero_division=0))
        train_weighted_f1 = float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0))
        val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        val_weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))

        train_cm = confusion_matrix(y_train, y_train_pred, labels=list(range(NUM_CLASSES))).tolist()
        val_cm = confusion_matrix(y_val, y_val_pred, labels=list(range(NUM_CLASSES))).tolist()

        val_acc_final  = float(val_acc_eval)
        val_loss_final = float(val_loss_eval)
        train_acc_final = float(train_acc)
        train_loss_final = float(train_loss)

        print(
            f"  [{self.client_id}] Round {server_round} | "
            f"val_acc: {val_acc_final:.4f} | val_loss: {val_loss_final:.4f} | "
            f"µ: {current_mu:.5f}"
        )

        return model.get_weights(), len(X_train), {
            "train_accuracy": train_acc_final,
            "train_loss": train_loss_final,
            "train_macro_f1": train_macro_f1,
            "train_weighted_f1": train_weighted_f1,
            "train_confusion_matrix": json.dumps(train_cm),
            "val_accuracy": val_acc_final,
            "val_loss": val_loss_final,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_confusion_matrix": json.dumps(val_cm),
            "client_id": self.client_id,
}

    def evaluate(self, parameters, config):
        """
        Evaluate global model on local test set.
        """
        X_test = self.client_data["X_test"]
        y_test = self.client_data["y_test"].astype(int)
        y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

        model = build_model(
            input_shape=self.input_shape,
            num_classes=NUM_CLASSES
        )
        model.set_weights(parameters)

        # ============================================================
        # INFERENCE TIME (ONLY ONE SAMPLE)
        # ============================================================
        n_experiments = 100
        # First sample for inference timing
        single_sample = X_test[0:1]
        # Warm-up (no counts)
        _ = model.predict(single_sample, verbose=0)
        inference_times_ms = []

        for _ in range(n_experiments):
            start_time = time.perf_counter()
            pred_single = model.predict(single_sample, verbose=0)
            end_time = time.perf_counter()
            inference_times_ms.append(
                (end_time - start_time) * 1000
            )

        mean_inference_ms = float(np.mean(inference_times_ms))
        std_inference_ms  = float(np.std(inference_times_ms))
        min_inference_ms  = float(np.min(inference_times_ms))
        max_inference_ms  = float(np.max(inference_times_ms))

        predicted_class = int(
            np.argmax(pred_single, axis=1)[0]
        )

        print(
            f"[{self.client_id}] "
            f"Inferencia (1 oveja): "
            f"{mean_inference_ms:.3f} ms "
            f"(std={std_inference_ms:.3f})"
        )

        # ============================================================
        # NORMAL EVALUTION ON TEST SET
        # ============================================================

        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        test_cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES))).tolist()
        test_report = classification_report(y_test,y_pred, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0,)

        print(
            f"[{self.client_id}] Eval | "
            f"acc: {test_acc:.4f} | "
            f"macro F1: {macro_f1:.4f}"
        )

        return float(test_loss), len(X_test), {
            "accuracy": float(test_acc),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": json.dumps(test_cm),
            "classification_report": json.dumps(test_report),
            "client_id": self.client_id,
            "inference_mean_ms": mean_inference_ms,
            "inference_std_ms": std_inference_ms,
            "inference_min_ms": min_inference_ms,
            "inference_max_ms": max_inference_ms,
            "predicted_class": predicted_class,
        }

# =============================================================================
# ENTRY POINT — used when running on Jetson Nano
# =============================================================================

def start_client(client_id, client_data, server_address, mu):
    """
    Launch a Flower client and connect to the server.
    Args:
        client_id (str): Client identifier.
        client_data (dict): Local tensors (X_train, y_train, X_val, y_val, X_test, y_test).
        server_address (str): Server IP
        mu (float): FedProx proximal coefficient.
    Return: None
    """
    client = SheepBehaviorClient(client_id=client_id, client_data=client_data, mu=mu)

    print(f"\n  Conectando al servidor: {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=client,)


# =============================================================================
# DIRECT EXECUTION — Jetson Nano
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower federated client for Jetson Nano")
    parser.add_argument("--client_id",  type=str, default="MureskDryPasture",
                        help="Client identifier (debe coincidir con la carpeta de datos)")
    parser.add_argument("--server_ip",  type=str, required=True,
                        help="IP del servidor (ordenador)")
    parser.add_argument("--server_port",type=str, default="8081",
                        help="Puerto del servidor (default: 8081)")
    parser.add_argument("--data_path",  type=str, default="data/clean",
                        help="Ruta a los datos del cliente")
    parser.add_argument("--mu",         type=float, default=MU,
                        help="Coeficiente proximal FedProx")
    args = parser.parse_args()

    # Add the parent directory to sys.path to import data_loader
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from data_loader import (
        load_federated_clients, prepare_client_splits, prepare_client_tensors
    )

    print(f"\n  Cargando datos de {args.client_id}...")
    # Load all clients data and prepare splits and tensors
    clients_data  = load_federated_clients(args.data_path)
    client_splits = prepare_client_splits(clients_data, random_seed=42)
    client_tensors = prepare_client_tensors(client_splits)

    if args.client_id not in client_tensors:
        raise ValueError(f"Cliente '{args.client_id}' no encontrado en {args.data_path}")

    server_address = f"{args.server_ip}:{args.server_port}"
    start_client(
        client_id=args.client_id,
        client_data=client_tensors[args.client_id],
        server_address=server_address,
        mu=args.mu
    )
