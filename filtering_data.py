import os
import numpy as np
import pandas as pd
import tkinter as tk 
from tkinter import filedialog
from utils import *


# def choose_output_folder(input_file_name, filterDataDict):
#         if "_a23" in input_file_name:
#             return os.path.join(filterDataDict, "a23")
#         elif "_f3" in input_file_name:
#             return os.path.join(filterDataDict, "f3")
#         else:
#             # Default output folder if no match is found
#             return "default_folder"
def main() :  
    folder_path = filedialog.askdirectory()
    # if os.path.exists(folder_path):
    #     # List all files in the folder
    #     file_list = os.listdir(folder_path)
    if folder_path:
        # Iterate through the files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path, sep=',')
        # Iterate through the files in the folder
        # for file_name in file_path:
        #     if file_name.endswith('.csv'):
        #         file_path = os.path.join(folder_path, file_name)
        #         df = pd.read_csv(file_path, sep=';')

                output_path = os.path.join(folder_path, file_path)

                modified_df = pd.read_csv(output_path)
                print("Modified CSV File Contents:")
                print(modified_df)

                #Step 6 calculate max, mean, median from csv 
                column_to_analyze = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D', 'Wifi_A_5G', 'Wifi_B_5G', 'Wifi_C_5G', 'Wifi_D_5G']

                 # Dictionary to store filtered data for each WiFi
                filtered_data = {}

                for column_name in column_to_analyze:
                    signal = modified_df[column_name].tolist()
                    print (signal)
                    signal_kalman_filter = kalman_filter(signal, A=1, H=1, Q=1.6, R=6)
                    # remove "array" in data
                    signal_kalman_filter = [item[0] if isinstance(item, np.ndarray) else item for item in signal_kalman_filter]
                    filtered_data[column_name] = signal_kalman_filter
                    # print("signal filtered for {0} : {1} ".format(column_name, signal_kalman_filter))
                    dfFilter = pd.DataFrame(signal_kalman_filter)
                    print("signal filtered for {0} : {1} ".format(column_name, dfFilter))
                    #define output folder for filtered data 
                
                # Create a DataFrame to store filtered data
                filtered_df = pd.DataFrame(filtered_data)
                
                # Create a folder 'filtered_output' if it doesn't exist
                output_folder = 'filtered_modified_output'
                output_folder_f3 = os.path.join(output_folder, "f3")
                output_folder_a23 = os.path.join(output_folder, "a23")
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)

                # Extract file name without extension
                file_name_without_ext = os.path.splitext(file_name)[0]

                # Save the filtered data to a CSV file in 'filtered_output' folder
                if ("_f3" in file_path or "_f23" in file_path or "_F3" in file_path):
                    output_file_path = os.path.join(output_folder_f3, f"{file_name_without_ext}_filtered.csv")
                    filtered_df.to_csv(output_file_path, index=False)
                    print(f"Filtered data from {file_name} saved to: {output_file_path}")
                elif("_a23" in file_path):
                    output_file_path = os.path.join(output_folder_a23, f"{file_name_without_ext}_filtered.csv")
                    filtered_df.to_csv(output_file_path, index=False)
                    print(f"Filtered data from {file_name} saved to: {output_file_path}")
                else:
                    print("wrong output directory")
                # # calculate and print mean, median, max 
                # for column_name in column_to_analyze:
                #     stats =calculate_statistics(modified_df, column_name)
                #     print(f"Column: {stats['Column']}")
                #     print(f"Mean: {stats['Mean']}")
                #     print(f"Median: {stats['Median']}")
                #     print(f"Maximum: {stats['Maximum']}")
                #     print()

                #     # Extract the statistics and column name
                #     mean_value = stats['Mean']
                #     median_value = stats['Median']
                #     max_value = stats['Maximum']

                #     # Create a new DataFrame for the statistics
                #     stats_df = pd.DataFrame({'Column': [column_name], 'Mean': [mean_value], 'Median': [median_value], 'Maximum': [max_value]})

                #     # Define the output file names
                #     # rawData = os.path.join(rawDataDict, f'Raw_statData_{column_name}.csv')
                #     rawData_folder = choose_output_folder(os.path.basename(output_path), filterDataFolder)
                #     print("rawdatafolder :",rawData_folder)
                #     # Check if the user canceled output folder selection
                #     if rawData_folder == "default_folder":
                #         print(f"Output folder for '{output_path}' selection canceled or not matched.")
                #     else:
                #         print(f"Output folder for '{output_path}' selected: {rawData_folder}")
                #         rawData = os.path.join(rawData_folder, f'Raw_statData_{column_name}.csv')
                #         print(rawData)

                #         # Append the statistics to the existing or new CSV file
                #         utils.append_to_csv(rawData, [stats], headers=['Column', 'Mean', 'Median', 'Maximum'])  

if __name__ == "__main__":
    main()
