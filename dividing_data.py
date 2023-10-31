import os
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split

def partition_data(df, num_partitions=10):
    partioned_data = []
    for _ in range(num_partitions):
        train_data, test_data = train_test_split(df, test_size=0.2)
        partioned_data.append((train_data, test_data))
    return partioned_data

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
                D = {}
                modified_df = pd.read_csv(output_path)
                print("Modified CSV File Contents:")
                print(modified_df)
                # Partition the data into Bayesian and Kalman datasets 10 times
                num_partitions = 10
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
