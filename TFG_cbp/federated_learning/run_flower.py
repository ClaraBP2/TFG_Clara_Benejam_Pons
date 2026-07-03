# -------------------------------------------------------------------------------------------------------------
# File Name                : run_flower.py
# Author                   : Clara Benejam Pons
# Description              : Run federated learning using Flower framework. Launches local clients as separate processes and starts the server.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------
 
import os
import sys
import time
import subprocess
 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import pandas as pd
import json
import flwr as fl
 
from data_loader import (
    load_federated_clients,
    summarize_clients,
    prepare_client_splits,
    summarize_client_splits,
    prepare_client_tensors,
    summarize_client_tensors,
)
from server_flower import FedProxStrategy, get_initial_parameters, NUM_ROUNDS
from model.clasification_model import build_model, NUM_CLASSES
 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
 
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================
 
CLEAN_DATA_PATH = "data/clean"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = "8081"
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"
 
LOCAL_CLIENTS = [
    "MurdochGreenPasture",
    "MureskBarley",
    "MureskStubble",
    "KatanningGreenPasture"
]
 
JETSON_CLIENT = "MureskDryPasture"
TOTAL_CLIENTS = len(LOCAL_CLIENTS) + 1

MU = 0.05 

# ============================================================
# MAIN
# ============================================================
 
if __name__ == "__main__":
 
    print("\n" + "=" * 70)
    print("FLOWER FEDERATED LEARNING — PC LAUNCHER")
    print("=" * 70)
    print(f"  Clientes locales : {LOCAL_CLIENTS}")
    print(f"  Cliente Jetson   : {JETSON_CLIENT}")
    print(f"  Servidor         : {SERVER_ADDRESS}")
    print("=" * 70)

    print("\n1. Comprobando datos")
    clients_data = load_federated_clients(CLEAN_DATA_PATH)
    summarize_clients(clients_data)
 
    client_splits = prepare_client_splits(clients_data, random_seed=42)
    summarize_client_splits(client_splits)
 
    client_tensors = prepare_client_tensors(client_splits)
    summarize_client_tensors(client_tensors)

    print("\n2. Lanzando clientes locales como procesos")
    processes = []
 
    for client_id in LOCAL_CLIENTS:
        if client_id not in client_tensors:
            print(f"  Warning: {client_id} no encontrado, saltando")
            continue
 
        cmd = [
            sys.executable,
            os.path.join("federated_learning", "client_flower.py"),
        ]
 
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"  Proceso lanzado: {client_id}")

    print(f"\n3. Lanzando servidor Flower en {SERVER_ADDRESS}...")
    print(f"   Esperando {TOTAL_CLIENTS} clientes: 4 PC + 1 Jetson")
    print("\n   *** En la Jetson ejecuta: ***")
    print(
        f"   python3 client_flower.py "
        f"--server_port {SERVER_PORT} "
        f"--client_id {JETSON_CLIENT}\n"
    )
 
    strategy = FedProxStrategy(
        mu=MU,
        local_epochs=4,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=TOTAL_CLIENTS,
        min_eval_clients=TOTAL_CLIENTS,
        min_available_clients=TOTAL_CLIENTS,
        initial_parameters=get_initial_parameters(),
    )
 
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config={"num_rounds": NUM_ROUNDS},
        strategy=strategy,
    )
 
    for p in processes:
        p.wait()
 
    if strategy.best_weights is not None:
        model = build_model(input_shape=(300, 4, 1), num_classes=NUM_CLASSES)
        model.set_weights(strategy.best_weights)
        os.makedirs("results/flower", exist_ok=True)
        model.save("results/flower/best_global_model.keras")
        print("\n  Mejor modelo guardado en: results/flower/best_global_model.keras")
        print(f"  Mejor val_acc: {strategy.best_val_acc:.4f}")
           
    metrics = []
 
    for row in strategy.round_history:
        metrics.append(row.copy())
 
    if hasattr(strategy, "evaluate_history"):
        for evaluate_row in strategy.evaluate_history:
            r = evaluate_row["round"]
            for row in metrics:
                if row["round"] == r:
                    row.update(evaluate_row)
 
    os.makedirs("results/flower", exist_ok=True)
 
    metrics_csv = f"results/flower/metrics_mu_{MU}.csv"
    metrics_json = f"results/flower/metrics_mu_{MU}.json"
    client_eval_csv = f"results/flower/client_eval_metrics_mu_{MU}.csv"
 
    if hasattr(strategy, "client_evaluate_history"):
        pd.DataFrame(strategy.client_evaluate_history).to_csv(client_eval_csv, index=False)
        print(f"  Métricas TEST por cliente guardadas en: {client_eval_csv}")

    if hasattr(strategy, "client_fit_history"):
        client_fit_csv = f"results/flower/client_train_validation_metrics_mu_{MU}.csv"
        pd.DataFrame(strategy.client_fit_history).to_csv(client_fit_csv, index=False)
        print(f"  Métricas TRAIN/VALIDATION por cliente guardadas en: {client_fit_csv}")

    if hasattr(strategy, "client_confusion_matrices"):
        cm_csv = f"results/flower/client_confusion_matrices_train_val_test_mu_{MU}.csv"
        pd.DataFrame(strategy.client_confusion_matrices).to_csv(cm_csv, index=False)
        print(f"  Matrices de confusión TRAIN/VAL/TEST guardadas en: {cm_csv}")
 
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
 
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=4)
 
    print(f"\n  Métricas guardadas en: {metrics_csv}")
    print(f"  Métricas guardadas en: {metrics_json}")

    best_row  = strategy.best_round_metrics
    best_rid  = best_row.get("round")
    best_eval = {}
    if hasattr(strategy, "evaluate_history"):
        for ev in strategy.evaluate_history:
            if ev["round"] == best_rid:
                best_eval = ev
                break

    summary = {
        "mu":                  MU,
        "rondas":              NUM_ROUNDS,
        "local_epochs":        strategy.local_epochs,
        "mean_tr_accuracy":    best_row.get("mean_train_accuracy"),
        "mean_tr_loss":        best_row.get("mean_train_loss"),
        "mean_val_accuracy":   best_row.get("mean_val_accuracy"),
        "mean_val_loss":       best_row.get("mean_val_loss"),
        "Macro_F1":            best_eval.get("mean_macro_f1"),
        "Epocas_locales":      strategy.local_epochs,
        "Mean_accuracy":       best_eval.get("mean_test_accuracy"),
        "Mean_Macro_F1":       best_eval.get("mean_macro_f1"),
        "Mean_Weighted_F1":    best_eval.get("mean_weighted_f1"),
        "best_round":          best_rid,
    }

    summary_csv = f"results/flower/summary_best_round_mu_{MU}.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"  Tabla resumen (mejor ronda) guardada en: {summary_csv}")
    print("\n  Entrenamiento federado completado.")
    
    
    from evaluation import run_local_baselines
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    print("\n" + "=" * 70)
    print("COMPARACIÓN: FEDERADO GLOBAL vs MODELOS INDIVIDUALES")
    print("=" * 70)
    
    results_dir = "results/flower/comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    global_model = build_model(input_shape=(300, 4, 1), num_classes=NUM_CLASSES)
    global_model.set_weights(strategy.best_weights)
    all_eval_tensors = dict(client_tensors)

    jetson_tensor_path = os.path.join(CLEAN_DATA_PATH, JETSON_CLIENT)
    if os.path.isdir(jetson_tensor_path):
        try:
            from data_loader import load_federated_clients, prepare_client_splits, prepare_client_tensors
            jetson_raw   = load_federated_clients(CLEAN_DATA_PATH, client_ids=[JETSON_CLIENT])
            jetson_split = prepare_client_splits(jetson_raw, random_seed=42)
            jetson_tens  = prepare_client_tensors(jetson_split)
            all_eval_tensors.update(jetson_tens)
            print(f"\n  Datos de la Jetson ({JETSON_CLIENT}) cargados para evaluación")
        except Exception as e:
            print(f"\n  No se pudieron cargar datos de la Jetson: {e}")
    else:
        print(f"\n  Directorio Jetson no encontrado ({jetson_tensor_path}), se omite del gráfico")

    fed_results = []

    for client_id, data in all_eval_tensors.items():
        X_test = data["X_test"]
        y_test = data["y_test"].astype(int)

        y_pred = np.argmax(global_model.predict(X_test, verbose=0), axis=1)

         pred_df = pd.DataFrame({
            "client_id": client_id,
            "y_real": y_test,
            "y_pred": y_pred
        })

        pred_df.to_csv(
            f"results/flower/predicciones_{client_id}.csv",
            index=False
        )

        fed_results.append({
            "client_id": client_id,
            "fed_accuracy": accuracy_score(y_test, y_pred),
            "fed_macro_f1": f1_score(y_test, y_pred, average="macro"),
            "fed_weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        })
    
    fed_df = pd.DataFrame(fed_results)
    
    baseline_df, baseline_models = run_local_baselines(
        all_eval_tensors,
        local_epochs=50,
        results_dir=results_dir,
    )

    local_df = baseline_df[[
        "client_id",
        "test_accuracy_sklearn",
        "macro_f1",
        "weighted_f1"
    ]].rename(columns={
        "test_accuracy_sklearn": "individual_accuracy",
        "macro_f1": "individual_macro_f1",
        "weighted_f1": "individual_weighted_f1",
    })
    
    comparison_df = pd.merge(fed_df, local_df, on="client_id")
    
    comparison_df["delta_accuracy"] = (
        comparison_df["fed_accuracy"] - comparison_df["individual_accuracy"]
    )
    
    comparison_df["delta_macro_f1"] = (
        comparison_df["fed_macro_f1"] - comparison_df["individual_macro_f1"]
    )
    
    comparison_path = os.path.join(
        results_dir,
        "comparison_federated_vs_individual.csv"
    )
    
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\nComparación por cliente:")
    print(comparison_df.to_string(index=False))
    
    print(f"\nCSV guardado en: {comparison_path}")
    
    clients = comparison_df["client_id"].tolist()
    x = np.arange(len(clients))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    def add_bar_labels(ax, bars):
        """Escribe el valor numérico encima de cada barra"""
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{height:.5f}",
                ha="center", va="bottom",
                fontsize=6.5, rotation=90,
            )

    bars_fed_acc = axes[0].bar(x - width / 2, comparison_df["fed_accuracy"], width, label="Federado")
    bars_ind_acc = axes[0].bar(x + width / 2, comparison_df["individual_accuracy"], width, label="Individual")
    add_bar_labels(axes[0], bars_fed_acc)
    add_bar_labels(axes[0], bars_ind_acc)
    axes[0].set_title("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clients, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.12)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    bars_fed_f1 = axes[1].bar(x - width / 2, comparison_df["fed_macro_f1"], width, label="Federado")
    bars_ind_f1 = axes[1].bar(x + width / 2, comparison_df["individual_macro_f1"], width, label="Individual")
    add_bar_labels(axes[1], bars_fed_f1)
    add_bar_labels(axes[1], bars_ind_f1)
    axes[1].set_title("Macro F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clients, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.12)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    plot_path = os.path.join(
        results_dir,
        "comparison_federated_vs_individual.png"
    )
    
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Gráfica guardada en: {plot_path}")
