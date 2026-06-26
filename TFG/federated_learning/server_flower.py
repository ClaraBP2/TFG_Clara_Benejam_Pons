# -------------------------------------------------------------------------------------------------------------
# File Name                : server_flower.py
# Author                   : Clara Benejam Pons
# Description              : Flower federated server with FedProx strategy for sheep behavior classification.
# -------------------------------------------------------------------------------------------------------------
 
import os
import sys
import json
import numpy as np
import pandas as pd
import flwr as fl
 
from typing import List, Tuple
from flwr.common import parameters_to_weights
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    weights_to_parameters,
    parameters_to_weights
)
 
from flwr.server.client_proxy import ClientProxy
 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
 
from model.clasification_model import build_model, NUM_CLASSES
from flwr.common import parameters_to_weights, weights_to_parameters
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
 
NUM_ROUNDS = 20
MIN_CLIENTS = 5
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8081
INPUT_SHAPE = (300, 4, 1)
RESULTS_DIR = "results/flower"
MU = 0.05 
LOCAL_EPOCHS = 4

# ============================================================
# FEDPROX STRATEGY
# ============================================================
 
class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx strategy.
    El término proximal mu se aplica en el cliente.
    El servidor agrega pesos con FedAvg ponderado y guarda métricas.
    """
 
    def __init__(self, mu: float = 0.05, local_epochs: int = 4, **kwargs):
        super().__init__(**kwargs)
 
        self.mu = mu
        self.local_epochs = local_epochs
 
        self.best_val_acc = -1.0
        self.best_val_macro_f1 = -1.0
        self.best_weights = None
        self.best_round_metrics = {}   # métricas completas de la mejor ronda
 
        self.round_history = []
        self.evaluate_history = []
        self.client_fit_history = []
        self.client_evaluate_history = []
        self.client_confusion_matrices = []
        
        
    def configure_fit(self, rnd, parameters, client_manager):
        config = {
            "server_round": rnd,
            "mu": self.mu,
            "local_epochs": self.local_epochs,
        }
 
        fit_ins = fl.common.FitIns(parameters, config)
 
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
 
        return [(client, fit_ins) for client in clients]
 
 
    def aggregate_fit(self,rnd: int,results: List[Tuple[ClientProxy, FitRes]],failures,):
        if not results:
            return None, {}

        # ----------------------------------------------------
        # 1. Agregación FedAvg ponderada
        # ----------------------------------------------------
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        num_examples_total = sum(num_examples for _, num_examples in weights_results)

        aggregated_weights = []

        for layer in range(len(weights_results[0][0])):
            layer_weights = [
                weights[layer] * num_examples
                for weights, num_examples in weights_results
            ]

            aggregated_layer = sum(layer_weights) / num_examples_total
            aggregated_weights.append(aggregated_layer)

        parameters_aggregated = weights_to_parameters(aggregated_weights)

        # ----------------------------------------------------
        # 2. Recoger métricas de entrenamiento/validación
        # ----------------------------------------------------
        val_accs = []
        val_losses = []
        val_macro_f1s = []
        val_weighted_f1s = []
        train_accs = []
        train_losses = []
        train_macro_f1s = []
        train_weighted_f1s = []

        for _, fit_res in results:
            metrics = fit_res.metrics or {}

            if "val_accuracy" in metrics:
                val_accs.append(float(metrics["val_accuracy"]))

            if "val_loss" in metrics:
                val_losses.append(float(metrics["val_loss"]))

            if "train_accuracy" in metrics:
                train_accs.append(float(metrics["train_accuracy"]))

            if "train_loss" in metrics:
                train_losses.append(float(metrics["train_loss"]))

            if "val_macro_f1" in metrics:
                val_macro_f1s.append(float(metrics["val_macro_f1"]))

            if "val_weighted_f1" in metrics:
                val_weighted_f1s.append(float(metrics["val_weighted_f1"]))

            if "train_macro_f1" in metrics:
                train_macro_f1s.append(float(metrics["train_macro_f1"]))

            if "train_weighted_f1" in metrics:
                train_weighted_f1s.append(float(metrics["train_weighted_f1"]))

            client_id = metrics.get("client_id", "unknown")
            self.client_fit_history.append({
                "round": rnd,
                "client_id": client_id,
                "num_examples": int(fit_res.num_examples),
                "train_loss": float(metrics.get("train_loss", 0.0)),
                "train_accuracy": float(metrics.get("train_accuracy", 0.0)),
                "train_macro_f1": float(metrics.get("train_macro_f1", 0.0)),
                "train_weighted_f1": float(metrics.get("train_weighted_f1", 0.0)),
                "val_loss": float(metrics.get("val_loss", 0.0)),
                "val_accuracy": float(metrics.get("val_accuracy", 0.0)),
                "val_macro_f1": float(metrics.get("val_macro_f1", 0.0)),
                "val_weighted_f1": float(metrics.get("val_weighted_f1", 0.0)),
            })

            for split_key, metric_key in [("train", "train_confusion_matrix"), ("validation", "val_confusion_matrix")]:
                if metric_key in metrics:
                    try:
                        self.client_confusion_matrices.append({
                            "round": rnd,
                            "client_id": client_id,
                            "split": split_key,
                            "confusion_matrix": metrics[metric_key],
                        })
                    except Exception:
                        pass

        mean_val_acc = float(np.mean(val_accs)) if val_accs else None
        mean_val_loss = float(np.mean(val_losses)) if val_losses else None
        mean_val_macro_f1 = float(np.mean(val_macro_f1s)) if val_macro_f1s else None
        mean_val_weighted_f1 = float(np.mean(val_weighted_f1s)) if val_weighted_f1s else None
        mean_train_acc = float(np.mean(train_accs)) if train_accs else None
        mean_train_loss = float(np.mean(train_losses)) if train_losses else None
        mean_train_macro_f1 = float(np.mean(train_macro_f1s)) if train_macro_f1s else None
        mean_train_weighted_f1 = float(np.mean(train_weighted_f1s)) if train_weighted_f1s else None

        # ----------------------------------------------------
        # 3. Guardar mejor modelo
        # ----------------------------------------------------
        if mean_val_macro_f1 is not None and mean_val_macro_f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = mean_val_macro_f1
            self.best_val_acc = mean_val_acc
            self.best_weights = aggregated_weights
            self.best_round_metrics = {
                "round": rnd,
                "mean_train_accuracy": mean_train_acc,
                "mean_train_loss": mean_train_loss,
                "mean_val_accuracy": mean_val_acc,
                "mean_val_loss": mean_val_loss,
                "mean_train_macro_f1": mean_train_macro_f1,
                "mean_train_weighted_f1": mean_train_weighted_f1,
                "mean_val_macro_f1": mean_val_macro_f1,
                "mean_val_weighted_f1": mean_val_weighted_f1,
            }
            print(f"\n  ✓ Nuevo mejor modelo global guardado | val_macro_f1={mean_val_macro_f1:.4f}")

        # ----------------------------------------------------
        # 4. Guardar historial por ronda
        # ----------------------------------------------------
        row = {
            "round": rnd,
            "num_clients_fit": len(results),
            "total_examples_fit": int(num_examples_total),
            "mean_train_accuracy": mean_train_acc,
            "mean_train_loss": mean_train_loss,
            "mean_val_accuracy": mean_val_acc,
            "mean_val_loss": mean_val_loss,
            "mean_train_macro_f1": mean_train_macro_f1,
            "mean_train_weighted_f1": mean_train_weighted_f1,
            "mean_val_macro_f1": mean_val_macro_f1,
            "mean_val_weighted_f1": mean_val_weighted_f1,
        }

        self.round_history.append(row)

        print(
            f"\n  → Round {rnd} FIT | "
            f"train_acc={mean_train_acc if mean_train_acc is not None else 'NA'} | "
            f"train_loss={mean_train_loss if mean_train_loss is not None else 'NA'} | "
            f"val_acc={mean_val_acc if mean_val_acc is not None else 'NA'} | "
            f"val_loss={mean_val_loss if mean_val_loss is not None else 'NA'} | "
            f"clients={len(results)}"
        )

        return parameters_aggregated, {
            "mean_train_accuracy": mean_train_acc if mean_train_acc is not None else 0.0,
            "mean_train_loss": mean_train_loss if mean_train_loss is not None else 0.0,
            "mean_val_accuracy": mean_val_acc if mean_val_acc is not None else 0.0,
            "mean_val_loss": mean_val_loss if mean_val_loss is not None else 0.0,
            "mean_train_macro_f1": mean_train_macro_f1 if mean_train_macro_f1 is not None else 0.0,
            "mean_train_weighted_f1": mean_train_weighted_f1 if mean_train_weighted_f1 is not None else 0.0,
            "mean_val_macro_f1": mean_val_macro_f1 if mean_val_macro_f1 is not None else 0.0,
            "mean_val_weighted_f1": mean_val_weighted_f1 if mean_val_weighted_f1 is not None else 0.0,
        }
 
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures,):
        if not results:
            return None, {}
 
        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
 
        weighted_loss = sum(
            evaluate_res.loss * evaluate_res.num_examples
            for _, evaluate_res in results
        ) / total_examples
 
        weighted_accuracy = sum(
            float(evaluate_res.metrics.get("accuracy", 0.0)) * evaluate_res.num_examples
            for _, evaluate_res in results
        ) / total_examples
 
        macro_f1s = [
            float(evaluate_res.metrics["macro_f1"])
            for _, evaluate_res in results
            if evaluate_res.metrics and "macro_f1" in evaluate_res.metrics
        ]
 
        weighted_f1s = [
            float(evaluate_res.metrics["weighted_f1"])
            for _, evaluate_res in results
            if evaluate_res.metrics and "weighted_f1" in evaluate_res.metrics
        ]
 
        mean_macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else None
        mean_weighted_f1 = float(np.mean(weighted_f1s)) if weighted_f1s else None
        min_macro_f1 = float(np.min(macro_f1s)) if macro_f1s else None
        std_macro_f1 = float(np.std(macro_f1s)) if macro_f1s else None

        min_weighted_f1 = float(np.min(weighted_f1s)) if weighted_f1s else None
        std_weighted_f1 = float(np.std(weighted_f1s)) if weighted_f1s else None
 
        for _, evaluate_res in results:
            metrics = evaluate_res.metrics or {}
 
            client_id = metrics.get("client_id", "unknown")
            self.client_evaluate_history.append({
                "round": server_round,
                "client_id": client_id,
                "num_examples": int(evaluate_res.num_examples),
                "test_loss": float(evaluate_res.loss),
                "test_accuracy": float(metrics.get("accuracy", 0.0)),
                "test_macro_f1": float(metrics.get("macro_f1", 0.0)),
                "test_weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                "classification_report": metrics.get("classification_report", ""),
            })
            if "confusion_matrix" in metrics:
                self.client_confusion_matrices.append({
                    "round": server_round,
                    "client_id": client_id,
                    "split": "test",
                    "confusion_matrix": metrics["confusion_matrix"],
                })
                
                
        row = {
            "round": server_round,
            "num_clients_eval": len(results),
            "total_examples_eval": int(total_examples),
            "mean_test_loss": float(weighted_loss),
            "mean_test_accuracy": float(weighted_accuracy),
            "mean_macro_f1": mean_macro_f1,
            "mean_weighted_f1": mean_weighted_f1,
            "min_macro_f1": min_macro_f1,
            "std_macro_f1": std_macro_f1,
            "min_weighted_f1": min_weighted_f1,
            "std_weighted_f1": std_weighted_f1,   
        }
 
        self.evaluate_history.append(row)
 
        print(
            f"  → Round {server_round} EVAL | "
            f"test_acc={weighted_accuracy:.4f} | "
            f"test_loss={weighted_loss:.4f} | "
            f"macro_f1={mean_macro_f1 if mean_macro_f1 is not None else 'NA'}"
        )
 
        return float(weighted_loss), {
            "mean_test_accuracy": float(weighted_accuracy),
            "mean_test_loss": float(weighted_loss),
            "mean_macro_f1": mean_macro_f1 if mean_macro_f1 is not None else 0.0,
            "mean_weighted_f1": mean_weighted_f1 if mean_weighted_f1 is not None else 0.0,
            "min_macro_f1": min_macro_f1 if min_macro_f1 is not None else 0.0,
            "std_macro_f1": std_macro_f1 if std_macro_f1 is not None else 0.0,
            "min_weighted_f1": min_weighted_f1 if min_weighted_f1 is not None else 0.0,
            "std_weighted_f1": std_weighted_f1 if std_weighted_f1 is not None else 0.0,
        }
    
    
 
 
# ============================================================
# INITIAL PARAMETERS
# ============================================================
 
def get_initial_parameters() -> Parameters:
    model = build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    weights = model.get_weights()
    return weights_to_parameters(weights)
 
 
# ============================================================
# SAVE METRICS
# ============================================================
 
def save_metrics(strategy: FedProxStrategy, results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
 
    metrics = []
 
    for row in strategy.round_history:
        metrics.append(row.copy())
 
    for evaluate_row in strategy.evaluate_history:
        round_id = evaluate_row["round"]
 
        matched = False
        for row in metrics:
            if row["round"] == round_id:
                row.update(evaluate_row)
                matched = True
                break
 
        if not matched:
            metrics.append(evaluate_row.copy())
 
    metrics_df = pd.DataFrame(metrics)
 
    csv_path  = os.path.join(results_dir, "metrics.csv")
    json_path = os.path.join(results_dir, "metrics.json")
 
    metrics_df.to_csv(csv_path, index=False)

    if getattr(strategy, "client_fit_history", None):
        pd.DataFrame(strategy.client_fit_history).to_csv(
            os.path.join(results_dir, "client_train_validation_metrics.csv"), index=False
        )

    if getattr(strategy, "client_evaluate_history", None):
        pd.DataFrame(strategy.client_evaluate_history).to_csv(
            os.path.join(results_dir, "client_test_metrics.csv"), index=False
        )

    if getattr(strategy, "client_confusion_matrices", None):
        pd.DataFrame(strategy.client_confusion_matrices).to_csv(
            os.path.join(results_dir, "client_confusion_matrices_train_val_test.csv"), index=False
        )
 
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
 
    print(f"\n  Métricas guardadas en: {csv_path}")
    print(f"  Métricas guardadas en: {json_path}")

    # ------------------------------------------------------------------
    # Tabla de resumen: mejor ronda por experimento
    # Columnas: mu | rondas | local_epochs | mean_tr_accuracy | mean_tr_loss
    #           | mean_val_accuracy | mean_val_loss | Macro_F1
    #           | Épocas locales | Mean accuracy | Mean Macro F1 | Mean Weighted F1
    # ------------------------------------------------------------------
    best_row = strategy.best_round_metrics  # métricas de la mejor ronda (val_macro_f1 más alta)

    # Buscar en evaluate_history la ronda que coincide con best_round
    best_eval = {}
    best_round_id = best_row.get("round")
    for ev in strategy.evaluate_history:
        if ev["round"] == best_round_id:
            best_eval = ev
            break

    summary = {
        "mu":                    strategy.mu,
        "rondas":                len(strategy.round_history),
        "local_epochs":          strategy.local_epochs,
        "mean_tr_accuracy":      best_row.get("mean_train_accuracy"),
        "mean_tr_loss":          best_row.get("mean_train_loss"),
        "mean_val_accuracy":     best_row.get("mean_val_accuracy"),
        "mean_val_loss":         best_row.get("mean_val_loss"),
        "Macro_F1":              best_eval.get("mean_macro_f1"),
        # Columnas nuevas (de la mejor ronda de evaluación)
        "Epocas_locales":        strategy.local_epochs,
        "Mean_accuracy":         best_eval.get("mean_test_accuracy"),
        "Mean_Macro_F1":         best_eval.get("mean_macro_f1"),
        "Mean_Weighted_F1":      best_eval.get("mean_weighted_f1"),
        "best_round":            best_round_id,
        
        "Min_Macro_F1":         best_eval.get("min_macro_f1"),
        "Std_Macro_F1":         best_eval.get("std_macro_f1"),
        "Min_Weighted_F1":      best_eval.get("min_weighted_f1"),
        "Std_Weighted_F1":      best_eval.get("std_weighted_f1"),
    }

    summary_df  = pd.DataFrame([summary])
    summary_csv = os.path.join(results_dir, "summary_best_round.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Tabla resumen (mejor ronda) guardada en: {summary_csv}")

    return metrics_df
 
 
# ============================================================
# RUN SERVER
# ============================================================
 
def run_server():
    strategy = FedProxStrategy(
        mu=MU,
        local_epochs=LOCAL_EPOCHS,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_eval_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=get_initial_parameters(),
    )
 
    print(f"\n{'=' * 70}")
    print("FLOWER FEDERATED SERVER")
    print(f"{'=' * 70}")
    print(f"  Host        : {SERVER_HOST}:{SERVER_PORT}")
    print(f"  Rounds      : {NUM_ROUNDS}")
    print(f"  Min clients : {MIN_CLIENTS}")
    print(f"  µ (FedProx) : {MU}")
    print(f"  Local epochs: {LOCAL_EPOCHS}")
    print(f"{'=' * 70}\n")
 
    fl.server.start_server(
    server_address=f"{SERVER_HOST}:{SERVER_PORT}",
    config={"num_rounds": NUM_ROUNDS},
    strategy=strategy,
)
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    # Guardar mejor modelo
    # ----------------------------------------------------
# Guardar mejor modelo global
# ----------------------------------------------------
    if hasattr(strategy, "best_weights") and strategy.best_weights is not None:

        try:
            model = build_model(
                input_shape=INPUT_SHAPE,
                num_classes=NUM_CLASSES
            )

            model.set_weights(strategy.best_weights)

            model_path = os.path.join(RESULTS_DIR, "best_global_model.keras")
            model.save(model_path)

            print(f"\n  ✓ Mejor modelo guardado en: {model_path}")
            print(f"  ✓ Mejor val_macro_f1 global: {strategy.best_val_macro_f1:.4f}")
            print(f"  ✓ Val_acc de esa ronda: {strategy.best_val_acc:.4f}")

        except Exception as e:
            print(f"\n  Error guardando el modelo: {e}")

    else:
        print("\n No se guardó modelo: best_weights es None")
        # Guardar métricas
    save_metrics(strategy, RESULTS_DIR)
    
    print("\n  Entrenamiento federado completado.")
    
    return strategy
 
 
if __name__ == "__main__":
    run_server()