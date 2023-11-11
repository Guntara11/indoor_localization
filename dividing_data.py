import os
import pandas as pd
import shutil
from scipy.spatial import distance
from scipy import stats
from scipy.optimize import minimize
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def partition_data(df, num_partitions=5):
    partioned_data = []
    for _ in range(num_partitions):
        train_data, test_data = train_test_split(df, test_size=0.2)
        partioned_data.append((train_data, test_data))
    return partioned_data

def plot_histogram(gaussian_histogram, title):
    plt.bar(range(len(gaussian_histogram)), gaussian_histogram, align='center')
    plt.title(title)
    plt.show()

def euclidean_distance(point1, point2):
    return distance.euclidean(point1, point2)

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

#memindahkan ke folder partition
def split_data (dataset_split_path, dataset_file, dataset_file_origin):
    if "partition_1" in dataset_file:
        dataset_folder_destination_satu = os.path.join(dataset_split_path, "partition_1")
        dataset_file_destination_satu = os.path.join(dataset_folder_destination_satu, dataset_file)

        if "partition_10" in dataset_file:
            dataset_folder_destination = os.path.join(dataset_split_path, "partition_10")
            dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

            shutil.move(dataset_file_origin, dataset_file_destination)
        else:
            shutil.move(dataset_file_origin, dataset_file_destination_satu)

    elif "partition_2" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_2")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_3" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_3")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_4" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_4")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_5" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_5")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    folder_path = filedialog.askdirectory()
    if os.path.exists(folder_path):
        # List all files in the folder
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, sep=';')
                output_path = os.path.join(folder_path, file_path)
                modified_df = pd.read_csv(output_path)
                print("Modified CSV File Contents:")
                print(modified_df)
                # Partition the data into Bayesian and Kalman datasets 10 times
                num_partitions = 5
                partitioned_data = partition_data(modified_df, num_partitions)
                # Print the sizes of the Bayesian and Kalman datasets for each partition
                for i, (train_data, test_data) in enumerate(partitioned_data, 1):
                    print(f"Partition {i}:")
                    print(f"Size of train Data (80%): {train_data.shape[0]} rows")
                    print(f"Size of test Data (20%): {test_data.shape[0]} rows")

                    # Save the Bayesian and Kalman data to separate CSV files for each partition
                    train_data.to_csv(f'dataset/train_{file_name}_partition_{i}.csv', index=False)
                    test_data.to_csv(f'dataset/test_{file_name}_partition_{i}.csv', index=False)
                print("Train Data (First Partition):")
                print(partitioned_data[0][0])

                print("Test Data (First Partition):")
                print(partitioned_data[0][1])

    #folder path dataset
    dataset_path = "dataset"

    #folder path dataset test dan train
    dataset_test_path = os.path.join(dataset_path, "test")
    dataset_train_path = os.path.join(dataset_path, "train")

    #folder path dataset test a23 dan f3
    dataset_test_a23_path = os.path.join(dataset_test_path, "a23")
    dataset_test_f3_path = os.path.join(dataset_test_path, "f3")

    #folder path dataset train a23 dan f3
    dataset_train_a23_path = os.path.join(dataset_train_path, "a23")
    dataset_train_f3_path = os.path.join(dataset_train_path, "f3")
    
    if os.path.exists(dataset_path):
        dataset_list_path = os.listdir(dataset_path)

    #iterasi list file dataset
    for dataset_file in dataset_list_path:
        
        #filter dataset test a23 (partisi 1 - 10)
        if "test_modified" in dataset_file and "_a23" in dataset_file:
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            split_data(dataset_split_path=dataset_test_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset test f3 (partisi 1 - 10)
        elif ("test_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_test_f3_path, dataset_file)
            split_data(dataset_split_path=dataset_test_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset train a23 (partisi 1 - 10)
        elif "train_modified" in dataset_file and "_a23" in dataset_file:
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_train_a23_path, dataset_file)
            split_data(dataset_split_path=dataset_train_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset train f3 (partisi 1 - 10)
        elif ("train_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_train_f3_path, dataset_file)
            split_data(dataset_split_path=dataset_train_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

####################################################################################################################
# Pindahkan code di bawah blok komen ke main py , dan juga buat proses di bawah untuk memproses semua data 
# yang ada di folder train dan test, 
# ##################################################################################################################
        # Load the train and test CSV files
    train_data = pd.read_csv("train_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")
    test_dataset = pd.read_csv("test_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")

    scaler = MinMaxScaler()
    test_dataset[["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]] = scaler.fit_transform(test_dataset[["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]])
    print("Specific Test Data (After Scaling):")
    print(test_dataset)

    # Specify the columns for Bayesian estimation
    stat_data = pd.read_csv("Raw_statData_Wifi_A.csv")
    centers = stat_data[["Mean", "Median", "Maximum"]].values
    # for i, (train_data, test_data) in enumerate(partitioned_data, 1):
    distances_mean = []
    distances_median = []
    distances_maximum = []

    for column in ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]:
        test_column = test_dataset[column]
        mean = stat_data.at[0, "Mean"]
        median = stat_data.at[0, "Median"]
        maximum = stat_data.at[0, "Maximum"]
        distances_mean.append(euclidean_distance(test_column, mean))
        distances_median.append(euclidean_distance(test_column, median))
        distances_maximum.append(euclidean_distance(test_column, maximum))


    # print(mean)
    # print(median)
    # print(maximum)
    # print(test_dataset)
    print(f"Euclidean Distances for Partition {i} (WiFi_A, WiFi_B, WiFi_C, WiFi_D):")
    print("Mean Distance:", distances_mean)
    print("Median Distance:", distances_median)
    print("Maximum Distance:", distances_maximum)

    
#   # Calculate RMSE between test_data and first Mean, Median, and Maximum for each column
#        # Calculate RMSE between test_data and first Mean, Median, and Maximum for each column
    # rmse_mean = [rmse(test_dataset[column], np.full(len(test_dataset), mean)) for column in ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]]
    # rmse_median = [rmse(test_dataset[column], np.full(len(test_dataset), median)) for column in ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]]
    # rmse_maximum = [rmse(test_dataset[column], np.full(len(test_dataset), maximum)) for column in ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]]



    # print("RMSE:")
    # print("Mean RMSE:", rmse_mean)
    # print("Median RMSE:", rmse_median)
    # print("Maximum RMSE:", rmse_maximum)
    # standard_deviation = 1.0  # Adjust the standard deviation as needed
    # probabilities_mean = [bayesian_likelihood_function(distance, standard_deviation) for distance in distances_mean]
    # probabilities_median = [bayesian_likelihood_function(distance, standard_deviation) for distance in distances_median]
    # probabilities_maximum = [bayesian_likelihood_function(distance, standard_deviation) for distance in distances_maximum]

    # print("Probabilities (WiFi_A, WiFi_B, WiFi_C, WiFi_D):")
    # print("Mean Probability:", probabilities_mean)
    # print("Median Probability:", probabilities_median)
    # print("Maximum Probability:", probabilities_maximum)





    # for i, (train_data, test_data) in enumerate(partitioned_data, 1):
    #     # Calculate Gaussian histograms for test data
    #     test_data_hist = hist_gauss(test_data["Wifi_A"])  # You can change the column as needed
    #     distances = []
    #     for center in centers.values:
    #         distance_val = euclidean_distance(test_data_hist, hist_gauss(center))
    #         distances.append(distance_val)
    #     print(f"Distance using Gaussian Histograms (Partition {i}):")
    #     print(distances)
    #     plot_histogram(test_data_hist, f'Gaussian Histogram for Test Data (Partition {i})')
    #     print(f"Gaussian Histogram for Test Data (Partition {i}):")
    #     print(test_data_hist)
    
    
    # Load test data
    # test_data = pd.read_csv("test_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")
    # train_data = pd.read_csv("train_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")
    # stat_data = pd.read_csv("Raw_statData_Wifi_A.csv")
    # # print("Mean Colum RP = ['Mean','Median','Maximum']
    # for target_column in AP:
    #     # Extract the target column for train and test data
    #     y_train = train_data[target_column]
    #     y_test = test_data[target_column]

    #     # Extract the selected features
    #     X_train = train_data[target_column]
    #     X_test = train_data[target_column]

    #     # Create and train the Bayesian Ridge model
    #     model = BayesianRidge()
    #     model.fit(X_train, y_train)

    #     # Perform Bayesian estimation on the test data
    #     y_pred = model.predict(X_test)

    #     # Calculate mean squared error as an example of performance evaluation
    #     mse = mean_squared_error(y_test, y_pred)
    #     print(f"Mean Squared Error for {target_column}: {mse}")

    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(y_test, y_pred, alpha=0.5)
    #     plt.xlabel("Actual Values")
    #     plt.ylabel("Predicted Values")
    #     plt.title(f"Bayesian Estimator for {target_column}")
    #     plt.grid(True)
    #     plt.show()n:")
    # # print(train_data["Mean"])
    # merged_test_data = test_data.join(stat_data.set_index(stat_data.index))
    # merged_train_data = test_data.join(stat_data.set_index(stat_data.index))
    # merged_test_data.to_csv("merged_test_data_a.csv", index=False)
    # merged_train_data.to_csv("merged_train_data_a.csv", index=False)

    # AP = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D']
    # #
                # for i in range(10):
                #     # shuffled_df.to_csv(f'dataset/shuffled_data_{file_name}.csv', index=False)
                #     trainData, testData = partition_data(modified_df)
                #     print(f"Size of Bayesian Data (80%): {trainData.shape[0]} rows")
                #     trainData.to_csv(f'dataset/train_data_{file_name}.csv', index=False)
                #     print(trainData)
                #     print(f"Size of Kalman Data (20%): {testData.shape[0]} rows")
                #     print(testData)
                # trainData.to_csv("train.csv", index=False)
                # testData.to_csv("test.csv", index=False)
