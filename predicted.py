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
    # print("Errors (Mean):", errors_mean)
    # print("Errors (Median):", errors_median)
    # print("Errors (Max):", errors_max)
    a23_error_mean_WifiA_2G = errors_mean['a23']['Wifi_A']['2.4GHz']
    a23_error_mean_WifiB_2G = errors_mean['a23']['Wifi_B']['2.4GHz']
    a23_error_mean_WifiC_2G = errors_mean['a23']['Wifi_C']['2.4GHz']
    a23_error_mean_WifiD_2G = errors_mean['a23']['Wifi_D']['2.4GHz']
    a23_error_mean_WifiA_5G = errors_mean['a23']['Wifi_A']['5GHz']
    a23_error_mean_WifiB_5G = errors_mean['a23']['Wifi_B']['5GHz']
    a23_error_mean_WifiC_5G = errors_mean['a23']['Wifi_C']['5GHz']
    a23_error_mean_WifiD_5G = errors_mean['a23']['Wifi_D']['5GHz']

    f3_error_mean_WifiA_2G = errors_mean['f3']['Wifi_A']['2.4GHz']
    f3_error_mean_WifiB_2G = errors_mean['f3']['Wifi_B']['2.4GHz']
    f3_error_mean_WifiC_2G = errors_mean['f3']['Wifi_C']['2.4GHz']
    f3_error_mean_WifiD_2G = errors_mean['f3']['Wifi_D']['2.4GHz']
    f3_error_mean_WifiA_5G = errors_mean['f3']['Wifi_A']['5GHz']
    f3_error_mean_WifiB_5G = errors_mean['f3']['Wifi_B']['5GHz']
    f3_error_mean_WifiC_5G = errors_mean['f3']['Wifi_C']['5GHz']
    f3_error_mean_WifiD_5G = errors_mean['f3']['Wifi_D']['5GHz']

    # print(a23_error_mean_WifiA_2G)
    # print(a23_error_mean_WifiB_2G)
    addlist1 = np.add(a23_error_mean_WifiA_2G, a23_error_mean_WifiB_2G)
    addlist2 = np.add(a23_error_mean_WifiC_2G, a23_error_mean_WifiD_2G)
    sumadd = np.add(addlist1, addlist2)
    # print(sumadd)

    #Average combinations device a23
    mean_error_combination1 = np.mean(np.vstack([a23_error_mean_WifiA_2G,a23_error_mean_WifiB_2G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error1 = np.mean(mean_error_combination1)
    mean_error_combination2 = np.mean(np.vstack([a23_error_mean_WifiA_2G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error2 = np.mean(mean_error_combination2)
    mean_error_combination3 = np.mean(np.vstack([a23_error_mean_WifiA_2G,a23_error_mean_WifiB_2G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error3 = np.mean(mean_error_combination3)
    mean_error_combination4 = np.mean(np.vstack([a23_error_mean_WifiA_2G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error4 = np.mean(mean_error_combination4)
    mean_error_combination5 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error5 = np.mean(mean_error_combination5)
    mean_error_combination6 = np.mean(np.vstack([a23_error_mean_WifiA_2G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error6 = np.mean(mean_error_combination6)
    mean_error_combination7 = np.mean(np.vstack([a23_error_mean_WifiA_2G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error7 = np.mean(mean_error_combination7)
    mean_error_combination8 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error8 = np.mean(mean_error_combination8)
    mean_error_combination9 = np.mean(np.vstack([a23_error_mean_WifiA_2G,a23_error_mean_WifiB_5G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error9 = np.mean(mean_error_combination9)
    mean_error_combination10 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error10 = np.mean(mean_error_combination10)
    mean_error_combination11 = np.mean(np.vstack([a23_error_mean_WifiA_5G,a23_error_mean_WifiB_5G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error11 = np.mean(mean_error_combination11)
    mean_error_combination12 = np.mean(np.vstack([a23_error_mean_WifiA_2G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error12 = np.mean(mean_error_combination12)
    mean_error_combination13 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_2G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error13 = np.mean(mean_error_combination13)
    mean_error_combination14 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_2G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error14 = np.mean(mean_error_combination14)
    mean_error_combination15 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_error15 = np.mean(mean_error_combination15)
    mean_error_combination16 = np.mean(np.vstack([a23_error_mean_WifiA_5G, a23_error_mean_WifiB_5G, a23_error_mean_WifiC_5G, a23_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_error16 = np.mean(mean_error_combination16)

    #Average combinations device f3
    mean_error_f3_combination1 = np.mean(np.vstack([f3_error_mean_WifiA_2G,f3_error_mean_WifiB_2G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error1 = np.mean(mean_error_f3_combination1)
    mean_error_f3_combination2 = np.mean(np.vstack([f3_error_mean_WifiA_2G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error2 = np.mean(mean_error_f3_combination2)
    mean_error_f3_combination3 = np.mean(np.vstack([f3_error_mean_WifiA_2G,f3_error_mean_WifiB_2G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error3 = np.mean(mean_error_f3_combination3)
    mean_error_f3_combination4 = np.mean(np.vstack([f3_error_mean_WifiA_2G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error4 = np.mean(mean_error_f3_combination4)
    mean_error_f3_combination5 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error5 = np.mean(mean_error_f3_combination5)
    mean_error_f3_combination6 = np.mean(np.vstack([f3_error_mean_WifiA_2G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error6 = np.mean(mean_error_f3_combination6)
    mean_error_f3_combination7 = np.mean(np.vstack([f3_error_mean_WifiA_2G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error7 = np.mean(mean_error_f3_combination7)
    mean_error_f3_combination8 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error8 = np.mean(mean_error_f3_combination8)
    mean_error_f3_combination9 = np.mean(np.vstack([f3_error_mean_WifiA_2G,f3_error_mean_WifiB_5G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error9 = np.mean(mean_error_f3_combination9)
    mean_error_f3_combination10 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error10 = np.mean(mean_error_f3_combination10)
    mean_error_f3_combination11 = np.mean(np.vstack([f3_error_mean_WifiA_5G,f3_error_mean_WifiB_5G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error11 = np.mean(mean_error_f3_combination11)
    mean_error_f3_combination12 = np.mean(np.vstack([f3_error_mean_WifiA_2G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error12 = np.mean(mean_error_f3_combination12)
    mean_error_f3_combination13 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_2G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error13 = np.mean(mean_error_f3_combination13)
    mean_error_f3_combination14 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_2G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error14 = np.mean(mean_error_f3_combination14)
    mean_error_f3_combination15 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_2G]), axis=0).tolist()
    average_mean_f3_error15 = np.mean(mean_error_f3_combination15)
    mean_error_f3_combination16 = np.mean(np.vstack([f3_error_mean_WifiA_5G, f3_error_mean_WifiB_5G, f3_error_mean_WifiC_5G, f3_error_mean_WifiD_5G]), axis=0).tolist()
    average_mean_f3_error16 = np.mean(mean_error_f3_combination16)
    
    #Error 95% a23
    percent_95_1 = np.percentile([mean_error_combination1],95)
    percent_95_2 = np.percentile([mean_error_combination2],95)
    percent_95_3 = np.percentile([mean_error_combination3],95)
    percent_95_4 = np.percentile([mean_error_combination4],95)
    percent_95_5 = np.percentile([mean_error_combination5],95)
    percent_95_6 = np.percentile([mean_error_combination6],95)
    percent_95_7 = np.percentile([mean_error_combination7],95)
    percent_95_8 = np.percentile([mean_error_combination8],95)
    percent_95_9 = np.percentile([mean_error_combination9],95)
    percent_95_10 = np.percentile([mean_error_combination10],95)
    percent_95_11 = np.percentile([mean_error_combination11],95)
    percent_95_12 = np.percentile([mean_error_combination12],95)
    percent_95_13 = np.percentile([mean_error_combination13],95)
    percent_95_14 = np.percentile([mean_error_combination14],95)
    percent_95_15 = np.percentile([mean_error_combination15],95)
    percent_95_16 = np.percentile([mean_error_combination16],95)
    
    #Error 95% f3
    percent_95_f3_1 = np.percentile([mean_error_f3_combination1],95)
    percent_95_f3_2 = np.percentile([mean_error_f3_combination2],95)
    percent_95_f3_3 = np.percentile([mean_error_f3_combination3],95)
    percent_95_f3_4 = np.percentile([mean_error_f3_combination4],95)
    percent_95_f3_5 = np.percentile([mean_error_f3_combination5],95)
    percent_95_f3_6 = np.percentile([mean_error_f3_combination6],95)
    percent_95_f3_7 = np.percentile([mean_error_f3_combination7],95)
    percent_95_f3_8 = np.percentile([mean_error_f3_combination8],95)
    percent_95_f3_9 = np.percentile([mean_error_f3_combination9],95)
    percent_95_f3_10 = np.percentile([mean_error_f3_combination10],95)
    percent_95_f3_11 = np.percentile([mean_error_f3_combination11],95)
    percent_95_f3_12 = np.percentile([mean_error_f3_combination12],95)
    percent_95_f3_13 = np.percentile([mean_error_f3_combination13],95)
    percent_95_f3_14 = np.percentile([mean_error_f3_combination14],95)
    percent_95_f3_15 = np.percentile([mean_error_f3_combination15],95)
    percent_95_f3_16 = np.percentile([mean_error_f3_combination16],95)
    
    #print Error
    print("rerata galat 1 :", average_mean_error1)
    print("rerata galat 2 :", average_mean_error2)
    print("rerata galat 3 :", average_mean_error3)
    print("rerata galat 4 :", average_mean_error4)
    print("rerata galat 5 :", average_mean_error5)
    print("rerata galat 6 :", average_mean_error6)
    print("rerata galat 7 :", average_mean_error7)
    print("rerata galat 8 :", average_mean_error8)
    print("rerata galat 9 :", average_mean_error9)
    print("rerata galat 10 :", average_mean_error10)
    print("rerata galat 11 :", average_mean_error11)
    print("rerata galat 12 :", average_mean_error12)
    print("rerata galat 13 :", average_mean_error13)
    print("rerata galat 14 :", average_mean_error14)
    print("rerata galat 15 :", average_mean_error15)
    print("rerata galat 16 :", average_mean_error16)
    #f3
    print("rerata galat f3 1 :", average_mean_f3_error1)
    print("rerata galat f3 2 :", average_mean_f3_error2)
    print("rerata galat f3 3 :", average_mean_f3_error3)
    print("rerata galat f3 4 :", average_mean_f3_error4)
    print("rerata galat f3 5 :", average_mean_f3_error5)
    print("rerata galat f3 6 :", average_mean_f3_error6)
    print("rerata galat f3 7 :", average_mean_f3_error7)
    print("rerata galat f3 8 :", average_mean_f3_error8)
    print("rerata galat f3 9 :", average_mean_f3_error9)
    print("rerata galat f3 10 :", average_mean_f3_error10)
    print("rerata galat f3 11 :", average_mean_f3_error11)
    print("rerata galat f3 12 :", average_mean_f3_error12)
    print("rerata galat f3 13 :", average_mean_f3_error13)
    print("rerata galat f3 14 :", average_mean_f3_error14)
    print("rerata galat f3 15 :", average_mean_f3_error15)
    print("rerata galat f3 16 :", average_mean_f3_error16)
    #95%
    print("percent 95 1 : ", percent_95_1)
    print("percent 95 2 : ", percent_95_2)
    print("percent 95 3 : ", percent_95_3)
    print("percent 95 4 : ", percent_95_4)
    print("percent 95 5 : ", percent_95_5)
    print("percent 95 6 : ", percent_95_6)
    print("percent 95 7 : ", percent_95_7)
    print("percent 95 8 : ", percent_95_8)
    print("percent 95 9 : ", percent_95_9)
    print("percent 95 10 : ", percent_95_10)
    print("percent 95 11 : ", percent_95_11)
    print("percent 95 12 : ", percent_95_12)
    print("percent 95 13 : ", percent_95_13)
    print("percent 95 14 : ", percent_95_14)
    print("percent 95 15 : ", percent_95_15)
    print("percent 95 16 : ", percent_95_16)
    #f3
    print("percent 95_f3_1 : ", percent_95_f3_1)
    print("percent 95_f3_2 : ", percent_95_f3_2)
    print("percent 95_f3_3 : ", percent_95_f3_3)
    print("percent 95_f3_4 : ", percent_95_f3_4)
    print("percent 95_f3_5 : ", percent_95_f3_5)
    print("percent 95_f3_6 : ", percent_95_f3_6)
    print("percent 95_f3_7 : ", percent_95_f3_7)
    print("percent 95_f3_8 : ", percent_95_f3_8)
    print("percent 95_f3_9 : ", percent_95_f3_9)
    print("percent 95_f3_10 : ", percent_95_f3_10)
    print("percent 95_f3_11 : ", percent_95_f3_11)
    print("percent 95_f3_12 : ", percent_95_f3_12)
    print("percent 95_f3_13 : ", percent_95_f3_13)
    print("percent 95_f3_14 : ", percent_95_f3_14)
    print("percent 95_f3_15 : ", percent_95_f3_15)
    print("percent 95_f3_16 : ", percent_95_f3_16)

    # Assuming you have defined these variables
    average_mean_errors = [average_mean_error1, average_mean_error2, average_mean_error3, average_mean_error4,
                        average_mean_error5, average_mean_error6, average_mean_error7, average_mean_error8,
                        average_mean_error9, average_mean_error10, average_mean_error11, average_mean_error12,
                        average_mean_error13, average_mean_error14, average_mean_error15, average_mean_error16]

    average_mean_f3_errors = [average_mean_f3_error1, average_mean_f3_error2, average_mean_f3_error3, average_mean_f3_error4,
                            average_mean_f3_error5, average_mean_f3_error6, average_mean_f3_error7, average_mean_f3_error8,
                            average_mean_f3_error9, average_mean_f3_error10, average_mean_f3_error11, average_mean_f3_error12,
                            average_mean_f3_error13, average_mean_f3_error14, average_mean_f3_error15, average_mean_f3_error16]

    percent_95_values = [percent_95_1, percent_95_2, percent_95_3, percent_95_4,
                        percent_95_5, percent_95_6, percent_95_7, percent_95_8,
                        percent_95_9, percent_95_10, percent_95_11, percent_95_12,
                        percent_95_13, percent_95_14, percent_95_15, percent_95_16]

    percent_95_f3_values = [percent_95_f3_1, percent_95_f3_2, percent_95_f3_3, percent_95_f3_4,
                            percent_95_f3_5, percent_95_f3_6, percent_95_f3_7, percent_95_f3_8,
                            percent_95_f3_9, percent_95_f3_10, percent_95_f3_11, percent_95_f3_12,
                            percent_95_f3_13, percent_95_f3_14, percent_95_f3_15, percent_95_f3_16]

    # Create a DataFrame
    data = {
        'Combination': [f'Combination {i}' for i in range(1, 17)],
        'Average Mean Error': average_mean_errors,
        'Average Mean Error f3': average_mean_f3_errors,
        '95th Percentile': percent_95_values,
        '95th f3 Percentile': percent_95_f3_values
    }

    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel('predict_skema1.xlsx', index=False)

if __name__ == "__main__":
    main()
