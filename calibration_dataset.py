import os
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import MinMaxScaler

# Function to read and process a single CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Columns to be calibrated using Min-Max scaling
    columns_to_scale = ['Wifi_A', 'Wifi_A_5G', 'Wifi_C', 'Wifi_B', 'Wifi_B_5G', 'Wifi_C_5G', 'Wifi_D', 'Wifi_D_5G']

    # Apply Min-Max scaling to specified columns
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    df = df.round(2)

    return df


# Folder paths
input_folders = {
    "train": [
        "calibrated_dataset/train/a23/partition_1",
        "calibrated_dataset/train/a23/partition_2",
        "calibrated_dataset/train/a23/partition_3",
        "calibrated_dataset/train/a23/partition_4",
        "calibrated_dataset/train/a23/partition_5",
        "calibrated_dataset/train/f3/partition_1",
        "calibrated_dataset/train/f3/partition_2",
        "calibrated_dataset/train/f3/partition_3",
        "calibrated_dataset/train/f3/partition_4",
        "calibrated_dataset/train/f3/partition_5",
        "filtered_calibrated_dataset/train/a23/partition_1",
        "filtered_calibrated_dataset/train/a23/partition_2",
        "filtered_calibrated_dataset/train/a23/partition_3",
        "filtered_calibrated_dataset/train/a23/partition_4",
        "filtered_calibrated_dataset/train/a23/partition_5",
        "filtered_calibrated_dataset/train/f3/partition_1",
        "filtered_calibrated_dataset/train/f3/partition_2",
        "filtered_calibrated_dataset/train/f3/partition_3",
        "filtered_calibrated_dataset/train/f3/partition_4",
        "filtered_calibrated_dataset/train/f3/partition_5"
    ],
    "test": [
        "calibrated_dataset/test/a23/partition_1",
        "calibrated_dataset/test/a23/partition_2",
        "calibrated_dataset/test/a23/partition_3",
        "calibrated_dataset/test/a23/partition_4",
        "calibrated_dataset/test/a23/partition_5",
        "calibrated_dataset/test/f3/partition_1",
        "calibrated_dataset/test/f3/partition_2",
        "calibrated_dataset/test/f3/partition_3",
        "calibrated_dataset/test/f3/partition_4",
        "calibrated_dataset/test/f3/partition_5",
        "filtered_calibrated_dataset/test/a23/partition_1",
        "filtered_calibrated_dataset/test/a23/partition_2",
        "filtered_calibrated_dataset/test/a23/partition_3",
        "filtered_calibrated_dataset/test/a23/partition_4",
        "filtered_calibrated_dataset/test/a23/partition_5",
        "filtered_calibrated_dataset/test/f3/partition_1",
        "filtered_calibrated_dataset/test/f3/partition_2",
        "filtered_calibrated_dataset/test/f3/partition_3",
        "filtered_calibrated_dataset/test/f3/partition_4",
        "filtered_calibrated_dataset/test/f3/partition_5"
    ]
}

for dataset_type, input_folder_list in input_folders.items():
    for input_folder in input_folder_list:
        # List all files in the input folder
        file_list = os.listdir(input_folder)

        # Process each CSV file in the input folder
        for file_name in file_list:
            if file_name.endswith(".csv"):
                input_file_path = os.path.join(input_folder, file_name)

                # Process the CSV file
                processed_data = process_csv(input_file_path)

                # Save the processed data to the output folder
                # output_file_path = os.path.join(input_folder, file_name)
                processed_data.to_csv(input_file_path, index=False)

print("Data calibration complete. Calibrated data saved in the 'calibrated' folder.")