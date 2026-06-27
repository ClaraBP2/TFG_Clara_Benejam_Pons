# -------------------------------------------------------------------------------------------------------------
# File Name                : clean_data.py
# Author                   : Clara Benejam Pons
# Description              : Data preparation for 3-class classification (active, grazing, inactive).
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

# ==============================================================================================================
# As known, the datasets have the following ORIGINAL classes: sitting, standing, walking, grazing and ruminating.
# 
# however, the final DESIRED classses are:
#    0 -> active (walking)
#    1 -> grazing (grazing)
#    2 -> inactive (sitting, standing, ruminating)
# ================================================================================================================

# Columns to keep
LABEL_COLUMNS = ["study_name", "sheep_number", "time_stamp","sitting", "standing", "walking", "grazing", "ruminating"]
# Class mapping for the new labels
CLASS_MAP = {
    "active": 0,
    "grazing": 1,
    "inactive": 2,
}


def clean_csv(file_path):
    """
    Function: Clean a single CSV file and generate 3-class labels.
    Arguments: file_path (str): Path to the raw CSV file.
    Return: Cleaned DataFrame with relevant columns and new 'label' and 'label_name' columns.
    """
    df = pd.read_csv(file_path, low_memory=False).copy()

    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    z_cols = [c for c in df.columns if c.startswith("z_")]

    accel_cols = x_cols + y_cols + z_cols
    for col in accel_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    behavior_cols = ["sitting", "standing", "walking", "grazing", "ruminating"]
    for col in behavior_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = pd.NA

    df.loc[df["grazing"] == 1, "label"] = CLASS_MAP["grazing"]
    df.loc[(df["walking"] == 1) & (df["label"].isna()), "label"] = CLASS_MAP["active"]
    inactive_mask = df[["ruminating", "standing", "sitting"]].eq(1).any(axis=1)
    df.loc[inactive_mask & (df["label"].isna()), "label"] = CLASS_MAP["inactive"]

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map({
        0: "active",
        1: "grazing",
        2: "inactive"
    })

    final_columns = LABEL_COLUMNS + x_cols + y_cols + z_cols + ["label", "label_name"]
    df = df[final_columns].copy()
    df = df.dropna().copy()

    return df


def process_file(input_path, output_path):
    """
    Function: Process a single file and save cleaned version.
    Arguments: input_path (str): Path to the raw CSV file.
               output_path (str): Path where the cleaned CSV will be saved.
    return: None (saves cleaned CSV to output_path)
    """
    print(f"Reading: {input_path}")
    df_clean = clean_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False, sep=";")

    print(f"Saved: {output_path} | rows: {len(df_clean)}")
    print("Class distribution:")
    print(df_clean["label_name"].value_counts())
    print("-" * 50)


def process_all_datasets(raw_base, clean_base):
    """
    Function: Process all folders and CSV files.
    Arguments: raw_base (str): Path to the base directory containing raw data.
               clean_base (str): Path to the base directory where cleaned data will be saved.
    Return: None (saves cleaned CSVs in the corresponding folders under clean_base)
    """
    folders = os.listdir(raw_base)

    for folder_name in folders:
        raw_folder_path = os.path.join(raw_base, folder_name)

        if not os.path.isdir(raw_folder_path):
            continue

        print(f"\nProcessing folder: {folder_name}")
        csv_files = glob.glob(os.path.join(raw_folder_path, "*.csv"))

        for csv_path in csv_files:
            file_name = os.path.basename(csv_path)
            output_name = file_name.replace(".csv", "_clean.csv")
            output_path = os.path.join(clean_base, folder_name, output_name)
            process_file(csv_path, output_path)


# This block allows the script to be run directly from the command line.
if __name__ == "__main__":
    raw_data_path = "data/raw"
    clean_data_path = "data/clean"
    process_all_datasets(raw_data_path, clean_data_path)
