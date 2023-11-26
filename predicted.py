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
    var = 0

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
                            # Initialize the array for Bayesian likelihood
                            bayesian_likelihood_function = np.zeros([7, 6, 3], dtype=float)
                            for axis_rp in range(7):
                                for ordinate_rp in range(6):
                                    mean_rp = mean_values_rp[device][beacon][frequency]["mean"][axis][ordinate]
                                    mean_td = statistics_td[device][beacon][frequency]["mean"][axis_rp][ordinate_rp]
                                    distance = np.sum(np.abs(mean_rp - mean_td) ** 2) / len(beacons)
                                    distances.append(distance)
                                    var += rssi_values_rp[ordinate_rp]
                                    var /= 4
                                    # Store or print the calculated distance
                                    distance_values[device][frequency][axis_rp][ordinate_rp] = np.sqrt(np.mean(distances))
                                    # Calculate Bayesian likelihood using the formula you provided
                                    bayesian_likelihood_function[axis_rp][ordinate_rp] = np.exp(-0.5 * (distance_values[device][frequency][axis_rp][ordinate_rp]) ** (1 / var))
                                print([beacon, device, frequency, bayesian_likelihood_function])
                            # print(rssi_values_rp)
                            # ic("RP data : ")
                            # ic(f"Data for {device_info}, {beacon}, {frequency}, ({axis}, {ordinate}):")
                            # ic(f"Mean : {mean_value_rp}, Median: {median_value_rp}, Max: {max_value_rp}")
                            # Print or store the calculated values
                            # print(f"Data for {device_info}, {beacon}, {frequency}, ({axis}, {ordinate}):")
                            # print(f"Mean: {mean_values}, Median: {median_values}, Max: {max_values}")
                        # print(len(rssi_rp_max))
                            # if device == "a23" and axis == 0 and ordinate == 0:
                            #     print(f"Data for {device_info}, {beacon}, {frequency}, ({axis}, {ordinate}) in test: \n {rssi_values}")
        else:
            print("path is no valid")
    # Print or use the calculated distance values
    # print(distance_values)
    
    


if __name__ == "__main__":
    main()





































# def main():
#     data_rp = "dataset/test/a23/partition_1"
#     data_td = "dataset/train/a23/partition_1"

#     wifi_signals = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D']

#     # Inisialisasi list untuk menyimpan nilai WiFi dari data_rp
#     Wifi_rp = {signal: [] for signal in wifi_signals}

#     # Menghitung metrik untuk data_rp
#     mean_array_rp, median_array_rp, max_array_rp = calculate_metrics(data_rp)

#     # Inisialisasi list untuk menyimpan nilai WiFi dari data_td
#     Wifi_td = {signal: [] for signal in wifi_signals}

#     # Menghitung metrik untuk data_td
#     mean_array_td, median_array_td, max_array_td = calculate_metrics(data_td)

#     # Read files from data_rp
#     for file_path_rp in os.listdir(data_rp):
#         if file_path_rp.startswith("test_modified"):
#             axis_rp, ordinate_rp = extract_axis_ordinate(file_path_rp)
#             print(axis_rp, ordinate_rp)

#             df = pd.read_csv(os.path.join(data_rp, file_path_rp))

#             # Extract WiFi values from the DataFrame
#             for signal in wifi_signals:
#                 Wifi_value = df[signal].values
#                 Wifi_rp[signal].append(Wifi_value)

#     # Read files from data_td
#     for file_path_td in os.listdir(data_td):
#         if file_path_td.startswith("train_modified"):
#             axis_td, ordinate_td = extract_axis_ordinate(file_path_td)
#             print(axis_td, ordinate_td)

#             df = pd.read_csv(os.path.join(data_td, file_path_td))

#             # Extract WiFi values from the DataFrame
#             for signal in wifi_signals:
#                 Wifi_value = df[signal].values
#                 Wifi_td[signal].append(Wifi_value)

#     # Convert lists to numpy arrays for data_rp
#     Wifi_rp_array = {signal: np.array(values) for signal, values in Wifi_rp.items()}

#     # Convert lists to numpy arrays for data_td
#     Wifi_td_array = {signal: np.array(values) for signal, values in Wifi_td.items()}

#     # Set print options for NumPy arrays
#     np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#     for signal in wifi_signals:
#         print(f'{signal} for data_rp:')
#         for values in Wifi_rp_array[signal]:
#             print(values)

#         print(f'{signal} for data_td:')
#         for values in Wifi_td_array[signal]:
#             print(values)


# import os
# import pandas as pd
# import numpy as np
# import re  
# import utils
# from utils import *

# def extract_axis_ordinate(filename):
#     # Extract axis and ordinate from the filename using a regular expression
#     match = re.search(r'\((\d+),(\d+)\)', filename)
#     if match:
#         axis, ordinate = map(int, match.groups())
#         return axis, ordinate
#     else:
#         return None, None
    
# def calculate_distance_power(mean_array_rp, mean_array_td):
#     return np.sum((abs(mean_array_td - mean_array_rp))**2) / 4

# def calculate_bayesian_likelihood(distance_power):
#     return np.exp(-0.5 * (distance_power**2) / 1)

# def main():
#     data_td = r'datasetNew/data_train/a23'
#     data_rp = r'datasetNew/data_test/a23'

#     # Menghitung metrik untuk data_rp
#     mean_array_rp, median_array_rp, max_array_rp = calculate_metrics(data_rp)

#     # Menghitung metrik untuk data_td
#     mean_array_td, median_array_td, max_array_td = calculate_metrics(data_td)

#     beacon_mapping = {0: "Wifi_A", 1: "Wifi_B", 2: "Wifi_C", 3: "Wifi_D"}
#     # Set the value for distance order and variance (you may need to adjust these values)
#     distance_order = 2
#     var = 1.0

#     bayesian_likelihood_mean = np.zeros((7, 6, 4))
#     bayesian_likelihood_median = np.zeros((7, 6, 4))
#     bayesian_likelihood_max = np.zeros((7, 6, 4))

#     # bayesian_likelihood_mean = np.empty([6, 7, 3], dtype=float)
#     # bayesian_likelihood_median = np.empty([6, 7, 3], dtype=float)
#     # bayesian_likelihood_max = np.empty([6, 7, 3], dtype=float)

#     for file_path in os.listdir(data_rp):
#         if file_path.startswith("test_modified"):
#             axis, ordinate = extract_axis_ordinate(file_path)
#             if axis is not None and ordinate is not None:
#                 for beacon in range(4):
#                     beacon_name = beacon_mapping[beacon]
#                     # Calculate distance_power for mean, median, and max
#                     distance_power_mean = calculate_distance_power(mean_array_rp[:, beacon], mean_array_td[:, beacon])
#                     distance_power_median = calculate_distance_power(median_array_rp[:, beacon], median_array_td[:, beacon])
#                     distance_power_max = calculate_distance_power(max_array_rp[:, beacon], max_array_td[:, beacon])

#                     # Calculate Bayesian likelihood
#                     bayesian_likelihood_mean[ordinate, axis, :] = calculate_bayesian_likelihood(distance_power_mean)
#                     bayesian_likelihood_median[ordinate, axis, :] = calculate_bayesian_likelihood(distance_power_median)
#                     bayesian_likelihood_max[ordinate, axis, :] = calculate_bayesian_likelihood(distance_power_max)

#                     print("axis: {0}, ordinate: {1}, beacon: {2}".format(axis, ordinate, beacon_name))
#                     print("Bayesian Likelihood (Mean):", bayesian_likelihood_mean)
#                     print("Bayesian Likelihood (Median):", bayesian_likelihood_median)
#                     print("Bayesian Likelihood (Max):", bayesian_likelihood_max)

# if __name__ == "__main__":
#     main()