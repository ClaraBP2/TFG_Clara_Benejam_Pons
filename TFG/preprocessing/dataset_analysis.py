# -------------------------------------------------------------------------------------------------------------
# File Name                : dataset_analysis.py
# Author                   : Clara Benejam Pons
# Creation Date            : 2026-03-30
# Description              : Clasificación de 3 clases (activa, pastando, inactiva) con CNN y validación LOSO
# Copyright                : (c) 2026 Clara Benejam Pons. All rights reserved.
# License                  : This code is private and may not be distributed without 
#                            explicit authorization from the author and the department.
#                            For academic or research use, please contact the author
#                            to request permission.
# Email                    : clara.benejam@alumnos.upm.es / vicente.hernandez@upm.es
# -------------------------------------------------------------------------------------------------------------

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["active", "grazing", "inactive"]

def compute_distribution(df):
    """"
    Function: compute the size and distribution of the classes in a given dataframe.
    Arguments: df (the input dataframe).
    Return: a dictionary with the number and percentage of samples for each class.
    """
    total = len(df)
    counts = df["label"].value_counts()
    pct = df["label"].value_counts(normalize=True) * 100

    distribution = {
        f"num_{name}": counts.get(i, 0)
        for i, name in {0: "active", 1: "grazing", 2: "inactive"}.items()
        } | {
        f"pct_{name}": round(pct.get(i, 0), 2)
        for i, name in {0: "active", 1: "grazing", 2: "inactive"}.items()
    }
    return distribution


def pie_chart(distribution, title, output_path):
    """
    Function: Create a pie chart showing the distribution of classes in the dataframe.
    Arguments: distribution (dict); A dictionary with the number of samples for each class.
               title (str); The title of the plot.
               output_path (str); The path where to save the plot.
    Return: None. Saves the plot to the specified path.
    """        
    values = [distribution[f"num_{l}"] for l in LABELS]

    filtered = [(l, v) for l, v in zip(LABELS, values) if v > 0]
    if not filtered:
        return

    labels, values = zip(*filtered)

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def plot_stacked_bar(df, value_type, title, output_path):
    """
    Function: Create a stacked bar chart showing the distribution of classes for each animal.
    Arguments: df (pandas.DataFrame); A dataframe with the distribution of classes for each animal.
                value_type (str); The type of value to plot (number or percentage).
                title (str); The title of the plot.
                output_path (str); The path where to save the plot.
    Return: None. Saves the plot to the specified path.
    """
    columns = [f"{value_type}_{l}" for l in LABELS]

    df.plot( x="animal", y=columns, kind="bar", stacked=True, figsize=(12, 6), title=title)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    

def process_study(study_folder, results_path):
    """
    Function: Process a single study folder, compute the distribution of classes for each animal, and create visualizations.
    Arguments: study_folder (str); The path to the study folder containing the CSV files for
               each animal.
               results_path (str); The path where to save the results (plots and distribution data).  
    Return: None. Saves the distribution data and plots to the specified path.
    """
    study_name = os.path.basename(study_folder)
    csv_files = glob.glob(os.path.join(study_folder, "*.csv"))

    if not csv_files:
        return

    os.makedirs(results_path, exist_ok=True)

    dfs = []
    rows = []

    for file in csv_files:
        df = pd.read_csv(file, sep=";")
        dfs.append(df)

        distribution = compute_distribution(df)

        rows.append({"animal": os.path.splitext(os.path.basename(file))[0],**distribution})

    df_all = pd.concat(dfs, ignore_index=True)
    distribution_all = compute_distribution(df_all)
    pie_chart(distribution_all, title=f"Global distribution - {study_name}", output_path=os.path.join(results_path, f"{study_name}_pie.png"))
    df_animals = pd.DataFrame(rows)
    plot_stacked_bar(df_animals, value_type="num", title=f"Counts - {study_name}", output_path=os.path.join(results_path, f"{study_name}_counts.png"))
    plot_stacked_bar(df_animals, value_type="pct",title=f"Percentages - {study_name}", output_path=os.path.join(results_path, f"{study_name}_pct.png"))


# This block allows the script to be run directly from the command line.
if __name__ == "__main__":
    clean_base = "data/clean"
    results_base = "results/dataset_analysis"

    study_folders = [
        os.path.join(clean_base, d)
        for d in os.listdir(clean_base)
        if os.path.isdir(os.path.join(clean_base, d))
    ]

    for study_folder in study_folders:
        study_name = os.path.basename(study_folder)
        results_path = os.path.join(results_base, study_name)
        print(f"Processing study: {study_name}")
        process_study(study_folder, results_path)