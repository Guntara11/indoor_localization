# import os
# import shutil

# #memindahkan ke folder partition
# def partition (dataset_split_path, dataset_file, dataset_file_origin):
#     if "partition_1" in dataset_file:
#         dataset_folder_destination_satu = os.path.join(dataset_split_path, "partition_1")
#         dataset_file_destination_satu = os.path.join(dataset_folder_destination_satu, dataset_file)

#         if "partition_10" in dataset_file:
#             dataset_folder_destination = os.path.join(dataset_split_path, "partition_10")
#             dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#             shutil.move(dataset_file_origin, dataset_file_destination)
#         else:
#             shutil.move(dataset_file_origin, dataset_file_destination_satu)

#     elif "partition_2" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_2")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_3" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_3")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_4" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_4")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_5" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_5")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_6" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_6")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_7" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_7")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_8" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_8")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)
        
#         shutil.move(dataset_file_origin, dataset_file_destination)

#     elif "partition_9" in dataset_file:
#         dataset_folder_destination = os.path.join(dataset_split_path, "partition_9")
#         dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

#         shutil.move(dataset_file_origin, dataset_file_destination)

# #folder path dataset
# dataset_path = "dataset"

# #folder path dataset test dan train
# dataset_test_path = os.path.join(dataset_path, "test")
# dataset_train_path = os.path.join(dataset_path, "train")

# #folder path dataset test a23 dan f3
# dataset_test_a23_path = os.path.join(dataset_test_path, "a23")
# dataset_test_f3_path = os.path.join(dataset_test_path, "f3")

# #folder path dataset train a23 dan f3
# dataset_train_a23_path = os.path.join(dataset_train_path, "a23")
# dataset_train_f3_path = os.path.join(dataset_train_path, "f3")

# if os.path.exists(dataset_path):
#     dataset_list_path = os.listdir(dataset_path)

#     #iterasi list file dataset
#     for dataset_file in dataset_list_path:
        
#         #filter dataset test a23 (partisi 1 - 10)
#         if "test_modified" in dataset_file and "_a23" in dataset_file:
#             dataset_file_origin = os.path.join(dataset_path, dataset_file)
#             partition(dataset_split_path=dataset_test_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

#         #filter dataset test f3 (partisi 1 - 10)
#         elif ("test_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
#             dataset_file_origin = os.path.join(dataset_path, dataset_file)
#             dataset_file_destination = os.path.join(dataset_test_f3_path, dataset_file)
#             partition(dataset_split_path=dataset_test_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

#         #filter dataset train a23 (partisi 1 - 10)
#         elif "train_modified" in dataset_file and "_a23" in dataset_file:
#             dataset_file_origin = os.path.join(dataset_path, dataset_file)
#             dataset_file_destination = os.path.join(dataset_train_a23_path, dataset_file)
#             partition(dataset_split_path=dataset_train_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

#         #filter dataset train f3 (partisi 1 - 10)
#         elif ("train_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
#             dataset_file_origin = os.path.join(dataset_path, dataset_file)
#             dataset_file_destination = os.path.join(dataset_train_f3_path, dataset_file)
#             partition(dataset_split_path=dataset_train_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

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

# ...

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

                # Menggunakan train_test_split langsung
                train_data, test_data = train_test_split(modified_df, test_size=0.2, random_state=42)

                print("Size of train Data (80%):", train_data.shape[0], "rows")
                print("Size of test Data (20%):", test_data.shape[0], "rows")

                # Menentukan folder penyimpanan
                train_folder = 'datasetNew/data_train'
                test_folder = 'datasetNew/data_test'

                # Membuat folder jika belum ada
                os.makedirs(train_folder, exist_ok=True)
                os.makedirs(test_folder, exist_ok=True)

                # Menentukan folder a23 dan f3
                a23_folder = 'a23'
                f3_folder = 'f3'

                # Membuat folder a23 dan f3 di dalam folder train dan test jika belum ada
                os.makedirs(os.path.join(train_folder, a23_folder), exist_ok=True)
                os.makedirs(os.path.join(train_folder, f3_folder), exist_ok=True)
                os.makedirs(os.path.join(test_folder, a23_folder), exist_ok=True)
                os.makedirs(os.path.join(test_folder, f3_folder), exist_ok=True)

                # Menentukan folder penyimpanan berdasarkan nama file
                if "_a23" in file_name:
                    train_folder = os.path.join(train_folder, a23_folder)
                    test_folder = os.path.join(test_folder, a23_folder)
                elif "_f3" in file_name or "_F3" in file_name or "_f23" in file_name :
                    train_folder = os.path.join(train_folder, f3_folder)
                    test_folder = os.path.join(test_folder, f3_folder)

                # Save the train and test data to separate CSV files
                train_data.to_csv(os.path.join(train_folder, f'train_{file_name}'), index=False)
                test_data.to_csv(os.path.join(test_folder, f'test_{file_name}'), index=False)

                print("Train Data:")
                print(train_data)

                print("Test Data:")
                print(test_data)
