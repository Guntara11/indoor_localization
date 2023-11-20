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

def calculate_distance_power(mean_array_rp, mean_array_td):
    return np.sum((abs(mean_array_td - mean_array_rp))**2) / mean_array_td.shape[0]

def calculate_bayesian_likelihood(distance_power):
    return np.exp(-0.5 * (distance_power**2) / 1)

def calculate_euclidean_distance(wifi_data, mean_array, metric='euclidean'):
    distances = np.zeros(wifi_data.shape[0])
    for i in range(wifi_data.shape[0]):
        for j in range(mean_array.shape[0]):
            distances[i, j] = distance.pdist([wifi_data.iloc[i].values, mean_array[j]], metric=metric)
    return distances

def main():
    data_rp = "datasetNew/data_test/a23"
    data_td = "datasetNew/data_train/a23"

    # Menghitung metrik untuk data_rp
    mean_array_rp, median_array_rp, max_array_rp = calculate_metrics(data_rp)
    # print("mean data :",mean_array_rp)
    # print("median data :",median_array_rp)
    # print("max data :",max_array_rp)
    # Inisialisasi list untuk menyimpan Bayesian likelihood
    all_bayesian_likelihood = []

    # # Menghitung metrik untuk data_td
    mean_array_td, median_array_td, max_array_td = calculate_metrics(data_td)
    beacon_mapping = {0: "Wifi_A", 1: "Wifi_B", 2: "Wifi_C", 3: "Wifi_D"}
    for file_path_rp in os.listdir(data_rp):
        if file_path_rp.startswith("test_modified"):
            axis_rp, ordinate_rp = extract_axis_ordinate(file_path_rp)
            print(axis_rp, ordinate_rp)
            if axis_rp is not None and ordinate_rp is not None:
                for file_path_td in os.listdir(data_td):
                    if file_path_td.startswith("train_modified"):
                        axis_td, ordinate_td = extract_axis_ordinate(file_path_td)
                        print("axis_td, ordinate_td")
                        print(axis_td, ordinate_td)
                        if axis_td is not None and ordinate_td is not None:
                            true_location = [axis_rp, ordinate_rp]
                            predictions = []
                            # Inisialisasi DataFrame untuk menyimpan Bayesian likelihood
                            bayesian_likelihood_df = pd.DataFrame(columns=['mean', 'median', 'max'])
                            for axis_rp in range(7):
                                for ordinate_rp in range(6):
                                    
                                    for beacon in range(4):
                                        beacon_name = beacon_mapping[beacon]

                                        # Calculate distance_power for mean, median, and max
                                        distance_power_mean = calculate_distance_power(mean_array_rp[:, beacon],
                                                                                        mean_array_td[:, beacon])
                                        distance_power_median = calculate_distance_power(median_array_rp[:, beacon],
                                                                                        median_array_td[:, beacon])
                                        distance_power_max = calculate_distance_power(max_array_rp[:, beacon],
                                                                                    max_array_td[:, beacon])
                                        # Calculate Bayesian likelihood
                                        bayesian_likelihood_mean = calculate_bayesian_likelihood(distance_power_mean)
                                        bayesian_likelihood_median = calculate_bayesian_likelihood(distance_power_median)
                                        bayesian_likelihood_max = calculate_bayesian_likelihood(distance_power_max)
                                        
                                        # Menambahkan ke DataFrame menggunakan loc
                                        bayesian_likelihood_df.loc[len(bayesian_likelihood_df)] = [bayesian_likelihood_mean, bayesian_likelihood_median, bayesian_likelihood_max]

                                        # Menambahkan ke list
                                        all_bayesian_likelihood.append([bayesian_likelihood_mean, bayesian_likelihood_median, bayesian_likelihood_max])

                                        # Mengonversi list menjadi ndarray
                                        combined_likelihood = np.array(all_bayesian_likelihood)[:, np.newaxis, :]

                                        # Menampilkan hasil array tiga dimensi
                                        print("Combined Bayesian Likelihood:")
                                        print(combined_likelihood)
                                    
                                        # # print("axis_rp: {0}, ordinate_rp: {1}, axis_td: {2}, ordinate_td: {3}, beacon: {4}".format(
                                        #     axis_rp, ordinate_rp, axis_td, ordinate_td, beacon_name))
                                        # print("Bayesian Likelihood (Mean):", bayesian_likelihood_mean)
                                        # print("Bayesian Likelihood (Median):", bayesian_likelihood_median)
                                        # print("Bayesian Likelihood (Max):", bayesian_likelihood_max)
                                    
                                # Mean
                                coordinate_mean = np.unravel_index(np.argmax(combined_likelihood[:, :, 0]),(6, 7))
                                prediction_mean = [coordinate_mean[1], coordinate_mean[0], math.sqrt((coordinate_mean[1] - ordinate_rp)**2 + (coordinate_mean[0] - axis_rp)**2)]
                                predictions.append(prediction_mean)

                                # Median
                                coordinate_median = np.unravel_index(np.argmax(combined_likelihood[:, :, 0]),(6, 7))
                                prediction_median = [coordinate_median[1], coordinate_median[0], math.sqrt((coordinate_median[1] - ordinate_rp)**2 + (coordinate_median[0] - axis_rp)**2)]
                                predictions.append(prediction_median)

                                # Max
                                coordinate_max = np.unravel_index(np.argmax(combined_likelihood[:, :, 0]),(6, 7))
                                prediction_max = [coordinate_max[1], coordinate_max[0], math.sqrt((coordinate_max[1] - ordinate_rp)**2 + (coordinate_max[0] - axis_rp)**2)]
                                predictions.append(prediction_max)

                                    # # Mean
                                    # coordinate_mean = np.unravel_index(np.argmax(combined_likelihood[:, :, 0]), (6, 7))
                                    # prediction_mean = [coordinate_mean[1], coordinate_mean[0], math.sqrt((coordinate_mean[1] - ordinate_rp)**2 + (coordinate_mean[0] - axis_rp)**2)]
                                    # predictions.append(prediction_mean)

                                    # # Median
                                    # coordinate_median = np.unravel_index(np.argmax(combined_likelihood[:, :, 1]), (6, 7))
                                    # prediction_median = [coordinate_median[1], coordinate_median[0], math.sqrt((coordinate_median[1] - ordinate_rp)**2 + (coordinate_median[0] - axis_rp)**2)]
                                    # predictions.append(prediction_median)

                                    # # Max
                                    # coordinate_max = np.unravel_index(np.argmax(combined_likelihood[:, :, 2]), (6, 7))
                                    # prediction_max = [coordinate_max[1], coordinate_max[0], math.sqrt((coordinate_max[1] - ordinate_rp)**2 + (coordinate_max[0] - axis_rp)**2)]
                                    # predictions.append(prediction_max)

                                print(coordinate_mean)
                                print(coordinate_median)
                                print(coordinate_max)
                                        
    # # Coordinates for rows
    # coordinates = [(i, j) for i in range(7) for j in range(6)]

    # # Beacon names for columns
    # beacon_names = ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]

    # # Create a DataFrame
    # data_dict = {"Coordinates": coordinates}
    # for i, beacon_name in enumerate(beacon_names):
    #     data_dict[beacon_name] = mean_array_rp[:, i]

    # df = pd.DataFrame(data_dict)

# Display the DataFrame
    # print(df)

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

