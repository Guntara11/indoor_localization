import os
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import MinMaxScaler

# Function to read and process a single CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(['TimeStampMillis', 'TimeStampFormatted'], axis=1)

    # Columns to be calibrated using Min-Max scaling
    columns_to_scale = ['Wifi_A', 'Wifi_A_5G', 'Wifi_C', 'Wifi_B', 'Wifi_B_5G', 'Wifi_C_5G', 'Wifi_D', 'Wifi_D_5G']

    # Apply Min-Max scaling to specified columns
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    df = df.round(2)

    return df

# Folder paths
input_folders = {
    "train": "modified_skema2/a23",
    "test": "modified_skema2/f3"
}
output_parent_folder = "calibrated_skema2"

for dataset_type in input_folders:
    output_folder = os.path.join(output_parent_folder, dataset_type)
    os.makedirs(output_folder, exist_ok=True)


for dataset_type, input_folder in input_folders.items():
    # List all files in the input folder
    file_list = os.listdir(input_folder)

    # Process each CSV file in the input folder
    for file_name in file_list:
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder, file_name)

            # Process the CSV file
            processed_data = process_csv(input_file_path)

            # Save the processed data to the output folder
            output_folder = os.path.join(output_parent_folder, dataset_type)
            output_file_path = os.path.join(output_folder, file_name)
            processed_data.to_csv(output_file_path, index=False)

print("Data calibration complete. Calibrated data saved in the 'calibrated' folder.")