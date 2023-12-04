import os
import pandas as pd
import numpy as np
import re  
import utils
from icecream import ic
from utils import *
from scipy.spatial import distance

def extract_file_info(filename):
    # Extract axis and ordinate from the filename
    match_axis_ordinate = re.search(r'\((\d+),(\d+)\)', filename)
    axis, ordinate = map(int, match_axis_ordinate.groups()) if match_axis_ordinate else (None, None)

    # Extract device information from the filename
    match_device_info = re.search(r'_([a-zA-Z0-9]+) -', filename)
    device_info = match_device_info.group(1) if match_device_info else None

    return device_info, axis, ordinate

def main():
    # Define the folder paths
    data_root = ""
    data_folder = os.path.join(data_root, "dataset")

    # Define the devices, beacons, and frequencies
    dataset = ["train", "test"]
    devices = ["a23", "f3"]
    beacons = ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]
    frequencies = ["2.4GHz", "5GHz"]

    # rssi_centering_list = []

    for data in dataset:
        if data == "train":
            dataset_folder = os.path.join(data_folder, "train")
            dataset_train_folder = dataset_folder

            for device in devices:
                device_folder = os.path.join(dataset_folder, device)
                partition_1_path = os.path.join(device_folder, "partition_1")
                rssi_centering_list = {beacon: {frequency: [] for frequency in frequencies} for beacon in beacons}
                for file_name in os.listdir(partition_1_path):
                    file_path = os.path.join(partition_1_path, file_name)
                    device_info, axis, ordinate = extract_file_info(file_name)
                    df_td = pd.read_csv(file_path)

                    for beacon in beacons:
                        for frequency in frequencies:
                            column_name = f"{beacon}"
                            if frequency == "5GHz":
                                column_name += "_5G"

                            rssi_values_td = df_td[column_name].tolist()
                            mean_value = np.mean(rssi_values_td)
                            median_value = np.median(rssi_values_td)
                            max_value = np.max(rssi_values_td)
                            variance_value = np.var(rssi_values_td)
                            # Append the results to the list
                            rssi_centering_list[beacon][frequency].append([mean_value, median_value, max_value, variance_value])
                            # print(f"Results for {beacon} - {frequency}:")
            for entry in rssi_centering_list[beacon][frequency]:
                print(entry)

                        # print(rssi_centering_point)
                            # print(mean_value)
                            # print(median_value)
                            # print(max_value)
                            # ic(device_info, axis, ordinate, beacon, frequency, mean_value, median_value, max_value)

        elif data == "test":
            dataset_folder = os.path.join(data_folder, "test")
            dataset_test_folder = dataset_folder

            for device in devices:
                device_folder = os.path.join(dataset_folder, device)
                partition_1_path = os.path.join(device_folder, "partition_1")

                for file_name in os.listdir(partition_1_path):
                    file_path = os.path.join(partition_1_path, file_name)
                    device_info, axis, ordinate = extract_file_info(file_name)
                    df_rp = pd.read_csv(file_path)

                    for beacon in beacons:
                        for frequency in frequencies:
                            column_name = f"{beacon}"
                            if frequency == "5GHz":
                                column_name += "_5G"

                            rssi_values_rp = df_rp[column_name].tolist()
                            # print(rssi_values_rp)
                            

if __name__ == "__main__":
    main()


