import os
import pandas as pd
import shutil
from scipy.spatial import distance
from scipy import stats
from scipy.optimize import minimize
from utils import *
from sklearn.model_selection import train_test_split

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


#memindahkan ke folder partition
def split_data (dataset_split_path, dataset_file, dataset_file_origin):
    if "partition_1" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_1")
        dataset_file_destination= os.path.join(dataset_folder_destination, dataset_file)
        shutil.move(dataset_file_origin, dataset_file_destination)

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
    if ("modified" in folder_path):
        if os.path.exists(folder_path):
            # List all files in the folder
            file_list = os.listdir(folder_path)
            print(file_list)
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
                        train_data.to_csv(f'calibrated_dataset/train_{file_name}_partition_{i}.csv', index=False)
                        test_data.to_csv(f'calibrated_dataset/test_{file_name}_partition_{i}.csv', index=False)
                    print("Train Data (First Partition):")
                    print(partitioned_data[0][0])

                    print("Test Data (First Partition):")
                    print(partitioned_data[0][1])

        #folder path dataset
        dataset_path = "dataset"
        dataset_calibration_path = "calibrated_dataset"

        #folder path dataset test dan train
        dataset_test_path = os.path.join(dataset_path, "test")
        dataset_train_path = os.path.join(dataset_path, "train")
        calibrated_dataset_test = os.path.join(dataset_calibration_path, "test")
        calibrated_dataset_train = os.path.join(dataset_calibration_path, "train")

        #folder path dataset test a23 dan f3
        dataset_test_a23_path = os.path.join(dataset_test_path, "a23")
        dataset_test_f3_path = os.path.join(dataset_test_path, "f3")
        calibrated_test_a23_path = os.path.join(calibrated_dataset_test, "a23")
        calibrated_test_f3_path = os.path.join(calibrated_dataset_test, "f3")

        #folder path dataset train a23 dan f3
        dataset_train_a23_path = os.path.join(dataset_train_path, "a23")
        dataset_train_f3_path = os.path.join(dataset_train_path, "f3")
        calibrated_train_a23_path = os.path.join(calibrated_dataset_train, "a23")
        calibrated_train_f3_path = os.path.join(calibrated_dataset_train, "f3")

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
    #     # Load the train and test CSV files
    # train_data = pd.read_csv("train_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")
    # test_dataset = pd.read_csv("test_modified (0,1)_a23 - 08-08-2023 14-10-13.csv_partition_1.csv")

    # scaler = MinMaxScaler()
    # test_dataset[["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]] = scaler.fit_transform(test_dataset[["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]])
    # print("Specific Test Data (After Scaling):")
    # print(test_dataset)

    # # Specify the columns for Bayesian estimation
    # stat_data = pd.read_csv("Raw_statData_Wifi_A.csv")
    # centers = stat_data[["Mean", "Median", "Maximum"]].values
    # # for i, (train_data, test_data) in enumerate(partitioned_data, 1):
    # distances_mean = []
    # distances_median = []
    # distances_maximum = []

    # for column in ["Wifi_A", "Wifi_B", "Wifi_C", "Wifi_D"]:
    #     test_column = test_dataset[column]
    #     mean = stat_data.at[0, "Mean"]
    #     median = stat_data.at[0, "Median"]
    #     maximum = stat_data.at[0, "Maximum"]
    #     distances_mean.append(euclidean_distance(test_column, mean))
    #     distances_median.append(euclidean_distance(test_column, median))
    #     distances_maximum.append(euclidean_distance(test_column, maximum))


        if os.path.exists(dataset_calibration_path):
            calibrated_dataset_list_path = os.listdir(dataset_calibration_path)

        #iterasi list file dataset
        for calibrated_dataset_file in calibrated_dataset_list_path:
            
            #filter dataset test a23 (partisi 1 - 10)
            if "test_modified" in calibrated_dataset_file and "_a23" in calibrated_dataset_file:
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_test_a23_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset test f3 (partisi 1 - 10)
            elif ("test_modified" in calibrated_dataset_file) and ("_f3" in calibrated_dataset_file or "_F3" in calibrated_dataset_file or "_f23" in calibrated_dataset_file):
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_test_f3_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_test_f3_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset train a23 (partisi 1 - 10)
            elif "train_modified" in calibrated_dataset_file and "_a23" in calibrated_dataset_file:
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_train_a23_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_train_a23_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset train f3 (partisi 1 - 10)
            elif ("train_modified" in calibrated_dataset_file) and ("_f3" in calibrated_dataset_file or "_F3" in calibrated_dataset_file or "_f23" in calibrated_dataset_file):
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_train_f3_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_train_f3_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)
    elif ("filtered" in folder_path):
        if os.path.exists(folder_path):
            # List all files in the folder
            file_list = os.listdir(folder_path)
            print(file_list)
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
                        train_data.to_csv(f'filtered_dataset/train_{file_name}_partition_{i}.csv', index=False)
                        test_data.to_csv(f'filtered_dataset/test_{file_name}_partition_{i}.csv', index=False)
                        train_data.to_csv(f'calibrated_filtered_dataset/train_{file_name}_partition_{i}.csv', index=False)
                        test_data.to_csv(f'calibrated_filtered_dataset/test_{file_name}_partition_{i}.csv', index=False)
                    print("Train Data (First Partition):")
                    print(partitioned_data[0][0])

                    print("Test Data (First Partition):")
                    print(partitioned_data[0][1])

        #folder path dataset
        dataset_path = "filtered_dataset"
        dataset_calibration_path = "calibrated_filtered_dataset"

        #folder path dataset test dan train
        dataset_test_path = os.path.join(dataset_path, "test")
        dataset_train_path = os.path.join(dataset_path, "train")
        calibrated_dataset_test = os.path.join(dataset_calibration_path, "test")
        calibrated_dataset_train = os.path.join(dataset_calibration_path, "train")

        #folder path dataset test a23 dan f3
        dataset_test_a23_path = os.path.join(dataset_test_path, "a23")
        dataset_test_f3_path = os.path.join(dataset_test_path, "f3")
        calibrated_test_a23_path = os.path.join(calibrated_dataset_test, "a23")
        calibrated_test_f3_path = os.path.join(calibrated_dataset_test, "f3")

        #folder path dataset train a23 dan f3
        dataset_train_a23_path = os.path.join(dataset_train_path, "a23")
        dataset_train_f3_path = os.path.join(dataset_train_path, "f3")
        calibrated_train_a23_path = os.path.join(calibrated_dataset_train, "a23")
        calibrated_train_f3_path = os.path.join(calibrated_dataset_train, "f3")

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



        if os.path.exists(dataset_calibration_path):
            calibrated_dataset_list_path = os.listdir(dataset_calibration_path)

        #iterasi list file dataset
        for calibrated_dataset_file in calibrated_dataset_list_path:
            
            #filter dataset test a23 (partisi 1 - 10)
            if "test_modified" in calibrated_dataset_file and "_a23" in calibrated_dataset_file:
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_test_a23_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset test f3 (partisi 1 - 10)
            elif ("test_modified" in calibrated_dataset_file) and ("_f3" in calibrated_dataset_file or "_F3" in calibrated_dataset_file or "_f23" in calibrated_dataset_file):
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_test_f3_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_test_f3_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset train a23 (partisi 1 - 10)
            elif "train_modified" in calibrated_dataset_file and "_a23" in calibrated_dataset_file:
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_train_a23_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_train_a23_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)

            #filter dataset train f3 (partisi 1 - 10)
            elif ("train_modified" in calibrated_dataset_file) and ("_f3" in calibrated_dataset_file or "_F3" in calibrated_dataset_file or "_f23" in calibrated_dataset_file):
                calibrated_dataset_file_origin = os.path.join(dataset_calibration_path, calibrated_dataset_file)
                calibrated_dataset_file_destination = os.path.join(calibrated_train_f3_path, calibrated_dataset_file)
                split_data(dataset_split_path=calibrated_train_f3_path, dataset_file=calibrated_dataset_file, dataset_file_origin=calibrated_dataset_file_origin)
