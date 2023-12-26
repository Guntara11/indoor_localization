import os
import pandas as pd
import numpy as np
import re  
import utils
from icecream import ic
from utils import *
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import csv
import itertools

trainingRatio = 0.8
trainingLength = int(trainingRatio*200)
windowLength = 8

distance_order = 2

dataset = ["train", "test"]
devices = ["a23", "f3"]
beacons = ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]
frequencies = ["2.4GHz", "5GHz"]

rssi_centering = []
output_file = 'hasil_prediksi.csv'

def extract_file_info(filename):
    # Extract axis and ordinate from the filename
    match_axis_ordinate = re.search(r'\((\d+),(\d+)\)', filename)
    axis, ordinate = map(int, match_axis_ordinate.groups()) if match_axis_ordinate else (None, None)

    # Extract device information from the filename
    match_device_info = re.search(r'_([a-zA-Z0-9]+) -', filename)
    device_info = match_device_info.group(1) if match_device_info else None

    return device_info, axis, ordinate


def calculate_metrics(rssi_values):
    mean_value = np.mean(rssi_values)
    median_value = np.median(rssi_values)
    max_value = np.max(rssi_values)
    variance_value = np.var(rssi_values)
    return mean_value, median_value, max_value, variance_value

def calculate_bayesian_likelihood(distance_power, variance, distance_order):
    return np.exp(-0.5 * (distance_power ** (2 / distance_order)) / variance)

def generate_combinations(beacons, frequencies):
    # Generate all possible combinations of beacons and frequencies
    combinations = list(itertools.product(beacons, frequencies))
    return combinations


def predict(train_data, test_data):
    errors_mean = {}
    errors_median = {}
    errors_max = {}
    for device in devices:
        # print(f"Device: {device}")
        errors_mean[device] = {}
        errors_median[device] = {}
        errors_max[device] = {}
        for beacon in beacons:
            errors_mean[device][beacon] = {}
            errors_median[device][beacon] = {}
            errors_max[device][beacon] = {}
            for frequency in frequencies:
                errors_mean[device][beacon][frequency] = []
                errors_median[device][beacon][frequency] = []
                errors_max[device][beacon][frequency] = []
                filtered_train_entries = [entry for entry in train_data if entry[0] == device and entry[3] == beacon and entry[4] == frequency]
                filtered_test_entries = [entry for entry in test_data if entry[0] == device and entry[3] == beacon and entry[4] == frequency]

                # Iterate over actual points
                for train_entry in filtered_train_entries:
                    axis, ordinate, rssi_values_td = train_entry[1], train_entry[2], train_entry[9]
                    # Access rssi_values_rp for each predicted point
                    predicted_points_mean = []
                    predicted_points_median = []  
                    predicted_points_max = []  
                    # Access rssi_values_rp for each predicted point
                    for test_entry in filtered_test_entries:
                        calc_axis, calc_ordinate, rssi_values_rp = test_entry[1], test_entry[2], test_entry[9]

                        # Calculate mean, median, max, and variance for rssi_values_td and rssi_values_rp
                        mean_value_td, median_value_td, max_value_td, variance_value_td = calculate_metrics(rssi_values_td)
                        mean_value_rp, median_value_rp, max_value_rp, variance_value_rp = calculate_metrics(rssi_values_rp)

                        mean_distance_power = np.sum(np.abs(mean_value_rp - mean_value_td) ** 2) / 4
                        median_distance_power = np.sum(np.abs(median_value_rp - median_value_td) ** 2) / 4
                        max_distance_power = np.sum(np.abs(max_value_rp - max_value_td) ** 2) / 4

                        # Calculate Bayesian likelihood values
                        likelihood_mean = calculate_bayesian_likelihood(mean_distance_power, variance_value_rp, distance_order)
                        likelihood_median = calculate_bayesian_likelihood(median_distance_power, variance_value_rp, distance_order)
                        likelihood_max = calculate_bayesian_likelihood(max_distance_power, variance_value_rp, distance_order)

                        predicted_points_mean.append((calc_axis, calc_ordinate, likelihood_mean))
                        predicted_points_median.append((calc_axis, calc_ordinate, likelihood_median))
                        predicted_points_max.append((calc_axis, calc_ordinate, likelihood_max))

                    best_predicted_point_mean = max(predicted_points_mean, key=lambda x: x[2])
                    best_predicted_point_median = max(predicted_points_median, key=lambda x: x[2])
                    best_predicted_point_max = max(predicted_points_max, key=lambda x: x[2])
                    
                    # Calculate error
                    error_mean = math.sqrt((best_predicted_point_mean[0] - axis) ** 2 + (best_predicted_point_mean[1] - ordinate) ** 2)
                    error_median = math.sqrt((best_predicted_point_median[0] - axis) ** 2 + (best_predicted_point_median[1] - ordinate) ** 2)
                    error_max = math.sqrt((best_predicted_point_max[0] - axis) ** 2 + (best_predicted_point_max[1] - ordinate) ** 2)
                    # Store errors separately for devices
                    # Store errors in the dictionaries
                    errors_mean[device][beacon][frequency].append(error_mean)
                    errors_median[device][beacon][frequency].append(error_median)
                    errors_max[device][beacon][frequency].append(error_max)
 
                    # print(f"Actual Point ({axis}, {ordinate}) - {beacon} ({frequency}):")
                    # print(f"Predicted Point (Based on Mean Likelihood): {best_predicted_point_mean[0]}, {best_predicted_point_mean[1]} - Error: {error_mean}")
                    # print(f"Predicted Point (Based on Median Likelihood): {best_predicted_point_median[0]}, {best_predicted_point_median[1]} - Error: {error_median}")
                    # print(f"Predicted Point (Based on Max Likelihood): {best_predicted_point_max[0]}, {best_predicted_point_max[1]} - Error: {error_max}")
    return errors_mean, errors_median, errors_max
                    

def main(): 
    # Define the folder paths
    data_root = ""
    data_folder = os.path.join(data_root, "dataset")

    train_data = []
    test_data = []
    for data in dataset:
        if data == "train":
            dataset_folder = os.path.join(data_folder, "train")
        elif data == "test":
            dataset_folder = os.path.join(data_folder, "test")

        for device in devices:
            device_folder = os.path.join(dataset_folder, device)
            partition_1_path = os.path.join(device_folder, "partition_1")

            for file_name in os.listdir(partition_1_path):
                file_path = os.path.join(partition_1_path, file_name)
                device_info, axis, ordinate = extract_file_info(file_name)
                df = pd.read_csv(file_path)

                for beacon in beacons:
                    for frequency in frequencies:
                        column_name = f"{beacon}"
                        if frequency == "5GHz":
                            column_name += "_5G"

                        # Use separate variables for train and test data
                        if data == "train":
                            rssi_values = df[column_name].tolist()
                            rssi_values_td = rssi_values
                            mean_value, median_value, max_value, variance_value = calculate_metrics(rssi_values_td)
                            train_data.append([device_info, axis, ordinate, beacon, frequency, mean_value, median_value, max_value, variance_value, rssi_values_td])
                            
                        elif data == "test":
                            rssi_values = df[column_name].tolist()
                            rssi_values_rp = rssi_values
                            mean_value, median_value, max_value, variance_value = calculate_metrics(rssi_values_rp)
                            test_data.append([device_info, axis, ordinate, beacon, frequency, mean_value, median_value, max_value, variance_value, rssi_values_rp])
                            # rssi_centering.append([mean_value, median_value, max_value, variance_value])
                            # print(rssi_centering)
                        # print(rssi_values_td)
                        # print(rssi_centering)
    errors_mean, errors_median, errors_max = predict(train_data, test_data)
    # Print or process the errors as needed
    print("Errors (Mean):", errors_mean)
    print("Errors (Median):", errors_median)
    print("Errors (Max):", errors_max)

if __name__ == "__main__":
    main()


