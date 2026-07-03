# -------------------------------------------------------------------------------------------------------------
# File Name                : evaluation.py
# Author                   : Clara Benejam Pons
# Description              : Evaluation utilities for federated learning, including local baselines, fine-tuning, confusion matrices, and comparison tables.
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
import pandas as pd
from federated_learning.client import train_local_baseline
from model.clasification_model import NUM_CLASSES, BATCH_SIZE

CLASS_NAMES = ["active", "grazing", "inactive"]
from federated_learning.client import get_class_weights
import tensorflow as tfjetson
import tensorflow as tf

FOCAL_LOSS = tf.keras.losses.CategoricalCrossentropy()
# =============================================================================
# 1. LOCAL BASELINE
# =============================================================================

def run_local_baselines(client_tensors, local_epochs=50, results_dir="results/evaluation"):
    """
    Train one independent model per client (no federation) and evaluate on test set.

    Args:
        client_tensors (dict): Output of prepare_client_tensors.
        local_epochs (int): Max epochs for local training (early stopping applies).
        results_dir (str): Directory to save results.

    Returns:
        baseline_results_df (pd.DataFrame): Test metrics per client.
        baseline_models (dict): Trained Keras models keyed by client_id.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("LOCAL BASELINE TRAINING")
    print("=" * 70)

    all_results = []
    baseline_models = {}

    for client_id, client_data in client_tensors.items():
        print(f"\n  Training local baseline for: {client_id}")

        model, history, results = train_local_baseline(
            client_id=client_id,
            client_data=client_data,
            local_epochs=local_epochs,
            loss=FOCAL_LOSS
        )

        baseline_models[client_id] = model

        print(
            f"  Epochs: {results['epochs_trained']} | "
            f"acc: {results['test_accuracy_sklearn']:.4f} | "
            f"macro F1: {results['macro_f1']:.4f}"
        )

        all_results.append(results)

    baseline_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("LOCAL BASELINE SUMMARY")
    print("=" * 70)
    print(baseline_df[["client_id", "test_accuracy_sklearn", "macro_f1", "weighted_f1", "epochs_trained"]])
    print(f"\n  Mean accuracy : {baseline_df['test_accuracy_sklearn'].mean():.4f} ± {baseline_df['test_accuracy_sklearn'].std():.4f}")
    print(f"  Mean macro F1 : {baseline_df['macro_f1'].mean():.4f} ± {baseline_df['macro_f1'].std():.4f}")

    baseline_df.to_csv(os.path.join(results_dir, "baseline_results2.csv"), index=False)
    print(f"\n  Saved: {os.path.join(results_dir, 'baseline_results2.csv')}")

    return baseline_df, baseline_models


# =============================================================================
# 2. CONFUSION MATRICES
# =============================================================================

def _get_predictions(model, X_test):
    return np.argmax(model.predict(X_test, verbose=0), axis=1)


def plot_confusion_matrix(ax, cm, title, class_names):
    """Plot a single normalized confusion matrix on a given axes."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicción", fontsize=8)
    ax.set_ylabel("Real", fontsize=8)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}\n({count})",
                    ha="center", va="center", fontsize=7, color=color)

    return im


def plot_all_confusion_matrices(
    global_model,
    baseline_models,
    client_tensors,
    finetuned_models,
    results_dir="results/evaluation"
):
    """
    Plot confusion matrices for federated global model, fine-tuned model, and local baseline,
    one row per client.

    Args:
        global_model: Trained federated global Keras model.
        baseline_models (dict): Local baseline models keyed by client_id.
        client_tensors (dict): Test data per client.
        finetuned_models (dict): Fine-tuned models keyed by client_id.
        results_dir (str): Output directory.
    """
    os.makedirs(results_dir, exist_ok=True)

    n_clients = len(client_tensors)
    fig, axes = plt.subplots(
        n_clients, 3,
        figsize=(15, 4 * n_clients),
        constrained_layout=True
    )

    # Handle single-client edge case
    if n_clients == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        "Matrices de confusión: modelo federado vs fine-tuned vs baseline local",
        fontsize=13, fontweight="bold", y=1.01
    )

    for row, (client_id, data) in enumerate(client_tensors.items()):
        X_test = data["X_test"]
        y_test = data["y_test"]

        y_pred_fed = _get_predictions(global_model, X_test)
        pred_df = pd.DataFrame({
            "client_id": [client_id] * len(y_pred_fed),
            "y_real": y_test,
            "y_pred": y_pred_fed
        })

        os.makedirs("results/flower", exist_ok=True)

        pred_df.to_csv(
            f"results/flower/predicciones_{client_id}.csv",
            index=False
        )
    
        cm_fed = confusion_matrix(y_test, y_pred_fed, labels=list(range(NUM_CLASSES)))
        acc_fed = accuracy_score(y_test, y_pred_fed)
        f1_fed = f1_score(y_test, y_pred_fed, average="macro")

        plot_confusion_matrix(
            axes[row, 0],
            cm_fed,
            f"{client_id}\nFederado · acc={acc_fed:.3f} · F1={f1_fed:.3f}",
            CLASS_NAMES
        )

        if client_id in finetuned_models:
            y_pred_ft = _get_predictions(finetuned_models[client_id], X_test)
            cm_ft = confusion_matrix(y_test, y_pred_ft, labels=list(range(NUM_CLASSES)))
            acc_ft = accuracy_score(y_test, y_pred_ft)
            f1_ft = f1_score(y_test, y_pred_ft, average="macro")

            plot_confusion_matrix(
                axes[row, 1],
                cm_ft,
                f"{client_id}\nFine-tuned · acc={acc_ft:.3f} · F1={f1_ft:.3f}",
                CLASS_NAMES
            )
        else:
            axes[row, 1].axis("off")
            axes[row, 1].text(0.5, 0.5, "Sin fine-tuned", ha="center", va="center")

        if client_id in baseline_models:
            y_pred_loc = _get_predictions(baseline_models[client_id], X_test)
            cm_loc = confusion_matrix(y_test, y_pred_loc, labels=list(range(NUM_CLASSES)))
            acc_loc = accuracy_score(y_test, y_pred_loc)
            f1_loc = f1_score(y_test, y_pred_loc, average="macro")

            plot_confusion_matrix(
                axes[row, 2],
                cm_loc,
                f"{client_id}\nLocal · acc={acc_loc:.3f} · F1={f1_loc:.3f}",
                CLASS_NAMES
            )
        else:
            axes[row, 2].axis("off")
            axes[row, 2].text(0.5, 0.5, "Sin baseline", ha="center", va="center")

    col_labels = ["Modelo Federado (FedProx)", "Fine-tuned", "Baseline Local"]
    for col, label in enumerate(col_labels):
        axes[0, col].annotate(
            label,
            xy=(0.5, 1.15), xycoords="axes fraction",
            ha="center", fontsize=11, fontweight="bold", color="#2E5597"
        )

    path = os.path.join(results_dir, "confusion_matrices2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices: {path}")


# =============================================================================
# 3. COMPARISON TABLE
# =============================================================================

def build_comparison_table(federated_results_df, finetuned_results_df,
                            baseline_results_df, results_dir="results/evaluation"):
    """
    Build a comparison table of federated vs fine-tuned vs local baseline results.
    """
    os.makedirs(results_dir, exist_ok=True)

    fed = federated_results_df[["client_id","test_accuracy_sklearn","macro_f1","weighted_f1"]].copy()
    fed.columns = ["client_id","fed_accuracy","fed_macro_f1","fed_weighted_f1"]

    ft = finetuned_results_df[["client_id","test_accuracy_sklearn","macro_f1","weighted_f1"]].copy()
    ft.columns = ["client_id","ft_accuracy","ft_macro_f1","ft_weighted_f1"]

    loc = baseline_results_df[["client_id","test_accuracy_sklearn","macro_f1","weighted_f1"]].copy()
    loc.columns = ["client_id","loc_accuracy","loc_macro_f1","loc_weighted_f1"]

    comp = pd.merge(pd.merge(fed, ft, on="client_id"), loc, on="client_id")

    comp["delta_fed_vs_loc"]  = comp["fed_accuracy"] - comp["loc_accuracy"]
    comp["delta_ft_vs_loc"]   = comp["ft_accuracy"]  - comp["loc_accuracy"]
    comp["delta_ft_vs_fed"]   = comp["ft_accuracy"]  - comp["fed_accuracy"]

    summary = {
        "client_id": "MEAN",
        "fed_accuracy":    comp["fed_accuracy"].mean(),
        "fed_macro_f1":    comp["fed_macro_f1"].mean(),
        "fed_weighted_f1": comp["fed_weighted_f1"].mean(),
        "ft_accuracy":     comp["ft_accuracy"].mean(),
        "ft_macro_f1":     comp["ft_macro_f1"].mean(),
        "ft_weighted_f1":  comp["ft_weighted_f1"].mean(),
        "loc_accuracy":    comp["loc_accuracy"].mean(),
        "loc_macro_f1":    comp["loc_macro_f1"].mean(),
        "loc_weighted_f1": comp["loc_weighted_f1"].mean(),
        "delta_fed_vs_loc": comp["delta_fed_vs_loc"].mean(),
        "delta_ft_vs_loc":  comp["delta_ft_vs_loc"].mean(),
        "delta_ft_vs_fed":  comp["delta_ft_vs_fed"].mean(),
    }
    comp = pd.concat([comp, pd.DataFrame([summary])], ignore_index=True)

    print("\n" + "=" * 70)
    print("FEDERATED vs FINE-TUNED vs LOCAL BASELINE COMPARISON")
    print("=" * 70)
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    path = os.path.join(results_dir, "comparison_federated_vs_finetuned_vs_local2.csv")
    comp.to_csv(path, index=False)
    print(f"\n  Saved comparison: {path}")

    return comp


def plot_comparison_bars(comparison_df, results_dir="results/evaluation"):
    """
    Plot side-by-side bar chart: federated vs local accuracy and macro F1 per client.
    """
    os.makedirs(results_dir, exist_ok=True)

    df = comparison_df[comparison_df["client_id"] != "MEAN"].copy()
    clients = df["client_id"].tolist()
    x = np.arange(len(clients))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Modelo federado vs baseline local por cliente", fontsize=13, fontweight="bold")

    for ax, metric, fed_col, loc_col, ylabel in [
        (axes[0], "Accuracy", "fed_accuracy", "loc_accuracy", "Test Accuracy"),
        (axes[1], "Macro F1", "fed_macro_f1", "loc_macro_f1", "Macro F1"),
    ]:
        bars_fed = ax.bar(x - width / 2, df[fed_col], width, label="Federado", color="#378ADD", alpha=0.9)
        bars_loc = ax.bar(x + width / 2, df[loc_col], width, label="Local",     color="#1D9E75", alpha=0.9)

        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("Green", "\nGreen").replace("Muresk", "Muresk\n") for c in clients], fontsize=8)
        ax.set_ylim(0.70, 1.00)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars_fed:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="#185FA5")
        for bar in bars_loc:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color="#0F6E56")

    path = os.path.join(results_dir, "comparison_bars2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison bars: {path}")


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def run_evaluation(global_model, final_results_df, client_tensors,
                   local_epochs=50, finetune_epochs=5, results_dir="results/evaluation"):
    """
    Full evaluation pipeline:
      1. Local baselines (entrenar modelo independiente por cliente)
      2. Fine-tuning del modelo global por cliente
      3. Confusion matrices (3 columnas: federado, fine-tuned, local)
      4. Tabla comparativa federado vs fine-tuned vs local
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n\n" + "=" * 70)
    print("EVALUATION: FEDERATED vs FINE-TUNED vs LOCAL BASELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Local baselines
    # ------------------------------------------------------------------
    baseline_df, baseline_models = run_local_baselines(
        client_tensors=client_tensors,
        local_epochs=local_epochs,
        results_dir=results_dir
    )

    # ------------------------------------------------------------------
    # 2. Fine-tuning del modelo global por cliente
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINE-TUNING GLOBAL MODEL PER CLIENT")
    print("=" * 70)

    finetuned_models = {}
    finetuned_results = []

    for client_id, data in client_tensors.items():
        print(f"\n  Fine-tuning for: {client_id}")
        ft_model = finetune_global_model(global_model, data, finetune_epochs)
        finetuned_models[client_id] = ft_model

        y_pred = np.argmax(ft_model.predict(data["X_test"], verbose=0), axis=1)
        acc = float(accuracy_score(data["y_test"], y_pred))
        f1m = float(f1_score(data["y_test"], y_pred, average="macro"))
        f1w = float(f1_score(data["y_test"], y_pred, average="weighted"))

        print(f"  acc: {acc:.4f} | macro F1: {f1m:.4f} | weighted F1: {f1w:.4f}")

        finetuned_results.append({
            "client_id": client_id,
            "test_accuracy_sklearn": acc,
            "macro_f1": f1m,
            "weighted_f1": f1w
        })

    finetuned_df = pd.DataFrame(finetuned_results)

    print("\n" + "=" * 70)
    print("FINE-TUNED SUMMARY")
    print("=" * 70)
    print(finetuned_df[["client_id", "test_accuracy_sklearn", "macro_f1", "weighted_f1"]])
    print(f"\n  Mean accuracy : {finetuned_df['test_accuracy_sklearn'].mean():.4f} ± {finetuned_df['test_accuracy_sklearn'].std():.4f}")
    print(f"  Mean macro F1 : {finetuned_df['macro_f1'].mean():.4f} ± {finetuned_df['macro_f1'].std():.4f}")

    finetuned_df.to_csv(os.path.join(results_dir, "finetuned_results2.csv"), index=False)
    print(f"\n  Saved: {os.path.join(results_dir, 'finetuned_results2.csv')}")

    # ------------------------------------------------------------------
    # 3. Confusion matrices (federado | fine-tuned | local)
    # ------------------------------------------------------------------
    print("\n  Generating confusion matrices...")
    plot_all_confusion_matrices(
        global_model=global_model,
        baseline_models=baseline_models,
        client_tensors=client_tensors,
        finetuned_models=finetuned_models,
        results_dir=results_dir, 
    )

    # ------------------------------------------------------------------
    # 4. Tabla comparativa: federado vs fine-tuned vs local
    # ------------------------------------------------------------------
    comp_df = build_comparison_table(
        federated_results_df=final_results_df,
        finetuned_results_df=finetuned_df,
        baseline_results_df=baseline_df,
        results_dir=results_dir
    )
    plot_comparison_bars(comp_df, results_dir=results_dir)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  All outputs saved to: {results_dir}/")
    print("    - baseline_results2.csv")
    print("    - finetuned_results2.csv")
    print("    - comparison_federated_vs_finetuned_vs_local2.csv")
    print("    - confusion_matrice2s.png")
    print("    - comparison_bar2s.png")

    return baseline_df, finetuned_df, comp_df

def finetune_global_model(global_model, client_data, finetune_epochs=5):
    """Fine-tuning local del modelo global para personalización."""
    
    local_model = tf.keras.models.clone_model(global_model)
    local_model.set_weights(global_model.get_weights())
    local_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss=FOCAL_LOSS,
        metrics=["accuracy"]
    )
    
    X_train = client_data["X_train"]
    y_train = to_categorical(client_data["y_train"], num_classes=NUM_CLASSES)
    class_weight = get_class_weights(client_data["y_train"])
    
    local_model.fit(
        X_train, y_train,
        epochs=finetune_epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=0
    )
    return local_model
