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
    # train_folder = os.path.join(data_root, "dataset", "train")
    # test_folder = os.path.join(data_root, "dataset", "test")

    # Define the devices, beacons, and frequencies
    dataset = ["train", "test"]
    devices = ["a23", "f3"]
    beacons = ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]
    frequencies = ["2.4GHz", "5GHz"]
    distanceOrder = 2

    statistics_td = {
        device: {
            beacon: {
                freq: {
                    "mean": [[None] * 6 for _ in range(7)],
                    "median": [[None] * 6 for _ in range(7)],
                    "max": [[None] * 6 for _ in range(7)],
                }
                for freq in frequencies
            }
            for beacon in beacons
        }
        for device in devices
    }
    statistics_rp = {
        device: {
            beacon: {
                freq: {
                    "mean": [[None] * 6 for _ in range(7)],
                    "median": [[None] * 6 for _ in range(7)],
                    "max": [[None] * 6 for _ in range(7)],
                }
                for freq in frequencies
            }
            for beacon in beacons
        }
        for device in devices
    }
    distances = {
        device: {
            freq: [[None] * 6 for _ in range(7)] for freq in frequencies
        }
        for device in devices
    }

    std_dev_values = {
        device: {
            beacon: {
                freq: [[None] * 6 for _ in range(7)] for freq in frequencies
            }
            for beacon in beacons
        }
        for device in devices
    }
    bayesian_likelihood_values = {
        device: {
            freq: [[None] * 6 for _ in range(7)] for freq in frequencies
        }
        for device in devices
    }
    mean_values_rp = statistics_td

    # Initialize a dictionary to store distance values
    distance_values = {device: {freq: [[None] * 6 for _ in range(7)] for freq in frequencies} for device in devices}

    for data in dataset:
        if data == "train":
            dataset_folder = os.path.join(data_folder, "train")
            dataset_train_folder = dataset_folder
            # print("data train path :",dataset_train_folder)
            for device in devices:
                device_folder = os.path.join(dataset_folder, device)
                partition_1_path = os.path.join(device_folder, "partition_1")
                # print("device_folder:", device_folder)
                
################tambahkan blok ini jika ingin memproses partisi 1 sampai partisi 5 untuk kebutuhan pengacakan data ############################

    #     # Loop over coordinates
                # for partition_folder in os.listdir(device_folder):
                # print("partition_folder:", partition_folder)
    # #         #     # Loop over data files in partition_1
                    # partition_1_path = os.path.join(device_folder, "partition_1")
                    # print(partition_1_path)
################################################################################################################################################
                for file_name in os.listdir(partition_1_path):
                    file_path = os.path.join(partition_1_path, file_name)
                    # print("file_path:", file_path)
                    # Extract device information, axis, and ordinate from the filename
                    device_info, axis, ordinate = extract_file_info(file_name)
                    # print("device_info:", device_info)
                    # print("axis:", axis, "ordinate:", ordinate)
                    df_td = pd.read_csv(file_path)
                    for beacon in beacons:
                        for frequency in frequencies:
                            column_name = f"{beacon}"
                            # print(column_name)
                            if frequency == "5GHz":
                                column_name += "_5G"
                                # print(column_name)
                            rssi_values_td = df_td[column_name].tolist()

                            mean_value_td = np.mean(rssi_values_td)
                            median_value_td = np.median(rssi_values_td)
                            max_value_td = np.max(rssi_values_td)
                            
                            statistics_td[device][beacon][frequency]["mean"][axis][ordinate] = mean_value_td
                            statistics_td[device][beacon][frequency]["median"][axis][ordinate] = median_value_td
                            statistics_td[device][beacon][frequency]["max"][axis][ordinate] = max_value_td


            # print(statistics_td)

                                    # if device == "a23" and axis == 0 and ordinate == 0:
                                    #     print(f"Data for {device_info}, {beacon}, {frequency}, ({axis}, {ordinate}) in train: \n {rssi_values}")
#################### blok RP data #############################        
        elif data == "test":
            dataset_folder = os.path.join(data_folder, "test")
            dataset_test_folder = dataset_folder
            # print("data test path :",dataset_test_folder)
            for device in devices:
                device_folder = os.path.join(dataset_folder, device)
                partition_1_path = os.path.join(device_folder, "partition_1")
                # print("device_folder:", device_folder)
################tambahkan blok ini jika ingin memproses partisi 1 sampai partisi 5 untuk kebutuhan pengacakan data ############################

    #     # Loop over coordinates
                # for partition_folder in os.listdir(device_folder):
                # print("partition_folder:", partition_folder)
    # #         #     # Loop over data files in partition_1
                    # partition_1_path = os.path.join(device_folder, "partition_1")
                    # print(partition_1_path)
################################################################################################################################################
                for file_name in os.listdir(partition_1_path):
                    file_path = os.path.join(partition_1_path, file_name)
                    # print("file_path:", file_path)
                    # Extract device information, axis, and ordinate from the filename
                    device_info, axis, ordinate = extract_file_info(file_name)
                    # print("device_info:", device_info)
                    # print("axis:", axis, "ordinate:", ordinate)
                    df_rp = pd.read_csv(file_path)
                    # print(df_rp)
                    for beacon in beacons:
                        for frequency in frequencies:
                            column_name = f"{beacon}"
                            # print(column_name)
                            if frequency == "5GHz":
                                column_name += "_5G"
                                # print(column_name)
                            rssi_values_rp = df_rp[column_name].tolist()
                            mean_value_rp = np.mean(rssi_values_rp)
                            median_value_rp = np.median(rssi_values_rp)
                            max_value_rp = np.max(rssi_values_rp)
                            # Calculate distance for each point
                            distances = []
                            for axis_rp in range(7):
                                for ordinate_rp in range(6):
                                    mean_rp = mean_values_rp[device][beacon][frequency]["mean"][axis][ordinate]
                                    mean_td = statistics_td[device][beacon][frequency]["mean"][axis_rp][ordinate_rp]
                                    distance = np.sum(np.abs(mean_rp - mean_td) ** 2) / len(beacons)
                                    distances.append(distance)
                                    distance_values[device][frequency][axis_rp][ordinate_rp] = np.sqrt(np.mean(distances))
                                    std_dev = np.std(rssi_values_td)
                                    std_dev_values[device][beacon][frequency][axis_rp][ordinate_rp] = std_dev
                                    bayesian_likelihood = np.exp(-((distance ** 2) / (2 * std_dev ** 2)))
                                    bayesian_likelihood_values[device][frequency][axis_rp][ordinate_rp] = bayesian_likelihood
                                    # print(f"Likelihood for ({axis_rp}, {ordinate_rp}): {likelihood}")
        else:
            print("path is no valid")
    # Print or use the calculated distance values
    # print(distance_values)
    # print(std_dev_values)
    print(bayesian_likelihood_values)
    
    


if __name__ == "__main__":
    main()


