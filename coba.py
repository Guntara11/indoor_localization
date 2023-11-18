import os
import pandas as pd
import numpy as np
import re  
import utils
from utils import *
from scipy.spatial import distance

def extract_axis_ordinate(filename):
    # Extract axis and ordinate from the filename using a regular expression
    match = re.search(r'\((\d+),(\d+)\)', filename)
    if match:
        axis, ordinate = map(int, match.groups())
        return axis, ordinate
    else:
        return None, None

def calculate_euclidean_distance(wifi_data, array_rp_point):
    distances = np.zeros(wifi_data.shape[0])
    for i in range(wifi_data.shape[0]):
        distances[i] = euclidean_distance(wifi_data.iloc[i].values, array_rp_point)
    return distances

def main():
    data_rp = r'C:\Users\User\Documents\Project\indoor_localization\datasetNew\data_test\a23'
    data_td = r'C:\Users\User\Documents\Project\indoor_localization\datasetNew\data_train\a23'

    # Menghitung metrik untuk data_rp
    mean_array_rp, median_array_rp, max_array_rp = calculate_metrics(data_rp)

    # Menghitung metrik untuk data_td
    mean_array_td, median_array_td, max_array_td = calculate_metrics(data_td)

    folder_path = data_td

    for filename in os.listdir(folder_path):
        if filename.startswith("train_modified_") and filename.endswith(".csv"):
            # Extract axis and ordinate from the filename
            axis, ordinate = extract_axis_ordinate(filename)
            if axis is not None and ordinate is not None:
                # Load the CSV file
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                wifi_columns = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D']
                wifi_data = df[wifi_columns]
                print("Extracted WiFi Data:")
                print(wifi_data)

                mean_array_rp_point = mean_array_rp[ordinate, axis*4:(axis+1)*4]
                median_array_rp_point = median_array_rp[ordinate, axis*4:(axis+1)*4]
                max_array_rp_point = max_array_rp[ordinate, axis*4:(axis+1)*4]

                # Debugging print statements
                print(f"Coordinate: ({axis}, {ordinate})")
                print("Wifi Data:")
                print(wifi_data)
                print("Mean Array RP Point:")
                print(mean_array_rp_point)
                print("Median Array RP Point:")
                print(median_array_rp_point)
                print("Max Array RP Point:")
                print(max_array_rp_point)
                
                distances_mean = calculate_euclidean_distance(wifi_data, mean_array_rp_point)
                distances_median = calculate_euclidean_distance(wifi_data, median_array_rp_point)
                distances_max = calculate_euclidean_distance(wifi_data, max_array_rp_point)
                # Display or store the results
                print(f"Coordinate: ({axis}, {ordinate})")
                print("Euclidean Distances (Mean):")
                print(distances_mean)

                print("\nEuclidean Distances (Median):")
                print(distances_median)

                print("\nEuclidean Distances (Max):")
                print(distances_max)


if __name__ == "__main__":
    main()

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

