import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utils
from utils import *
"""
'''
Author: Ega Guntara
Date: 23/09/2023
Description: Brief description of the code's purpose
'''  
"""

import numpy as np
from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt

# # Data points and coordinates
# data = [-50.55, -45.7, -40.8, -46.6, -46.25, -43.15, -46.1, 
#         -10.55, -25.7, -30.8, -46.6, -76.25, -13.15, -26.1]
# coordinates = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
#                (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), 
#                (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]

# # Create a 2D grid for the heatmap
# heatmap = np.zeros((7, 3))  # Assuming 7 rows and 4 columns

# # Populate the grid with data at the specified coordinates
# for (x, y), value in zip(coordinates, data):
#     heatmap[x, y] = value

# # Create the heatmap plot
# plt.imshow(heatmap, cmap='viridis', aspect='auto')
# plt.colorbar()

# # Label the axes
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')

# # Show the heatmap
# plt.show()

"""
COBA KODE DI BAWAH , untuk 42 titik 
hapus bagian data 
"""

#################################################### part untuk di hapus atau di comment aja ################################################################
# data = [
#     [-50.55, -50.0, -47.0],
#     [-45.7, -46.0, -43.0],
#     [-40.8, -40.5, -38.0],
#     [-46.6, -47.0, -39.0],
#     [-46.25, -46.0, -43.0],
#     [-43.15, -43.0, -40.0],
#     [-46.1, -46.0, -37.0]
# ]
##############################################################################################################################################################
# def make_visualization(data):
#     # Extract the first items from each sublist
#     first_items = [sublist[0] for sublist in data]
#     # data_type = type(first_items)
#     # print(data_type)
#     # print(first_items)

#     # Create a correlation matrix
#     correlation_matrix = np.zeros((len(data), len(data)))

#     for i in range(len(data)):
#         for j in range(len(data)):
#             # Calculate the Pearson correlation coefficient between the first items at positions i and j
#             correlation, _ = pearsonr(data[i], data[j])
#             correlation_matrix[i][j] = correlation

#     # Print the correlation matrix
#     print(correlation_matrix)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=range(1, 8), yticklabels=range(1, 5))
#     plt.title("Correlation Heatmap")
#     plt.show()
#################################################### part untuk di hapus atau di comment aja ################################################################
"""
SESUAIKAN AXIS X DAN Y SESUAI BANYAK NYA KOORDINATE YANG TERCANTUM DI FILE MODIFY 
CONTOH (0,1) ADALAH X = 0 DAN Y =1 
"""
##############################################################################################################################################################

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    data_dict = {}  # Store data for different Wi-Fi networks
    wifi_name_list = []  # Store Wi-Fi network names

    folder_path = filedialog.askdirectory()
    if os.path.exists(folder_path):
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_name = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_name, sep=',')
            wifi_name = df.loc[0, 'Column']

            if wifi_name not in data_dict:
                data_dict[wifi_name] = {'Mean': [], 'Median': [], 'Max': []}  # Initialize lists for the Wi-Fi network

            nested_data = df[['Mean', 'Median', 'Maximum']].values.tolist()
            data_dict[wifi_name]['Mean'].append([sublist[0] for sublist in nested_data])
            data_dict[wifi_name]['Median'].append([sublist[1] for sublist in nested_data])
            data_dict[wifi_name]['Max'].append([sublist[2] for sublist in nested_data])
            wifi_name_list.append(wifi_name)

    # # make_visualization(nested_data)//untuk 7 titik
    # y_coordinates =  [3] * len(data_dict[wifi_name_list[0]]['Mean'][0]) 
    # # Create X-coordinates for the data points (0 to 6)
    # x_coordinates = list(range(len(data_dict[wifi_name_list[0]]['Mean'][0])))

    # Create X-coordinates and Y-Coordinates for 42 points
    num_columns = 7  

    x_coordinates = [i % num_columns for i in range(len(data_dict[wifi_name_list[0]]['Mean'][0]))]
    y_coordinates = [i // num_columns for i in range(len(data_dict[wifi_name_list[0]]['Mean'][0]))]
    
    num_plots = len(wifi_name_list)
    cols = num_plots
    rows = 3

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10), gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

    for i, wifi_name in enumerate(wifi_name_list):
        for j, metric in enumerate(['Mean', 'Median', 'Max']):
            data_matrix = data_dict[wifi_name][metric]

            ax = axes[j, i]
            im = ax.scatter(x_coordinates, y_coordinates, c=data_matrix, cmap='YlGnBu', s=100, marker='s')  # Use 's' for square markers
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'{metric}\n {wifi_name}')
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')

    # Create the colorbar and specify the axes to use
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, label='Values')
    plt.show()


    # if nested_data:
    #         # Process and work with the nested_data as needed in your main script
    #         print("Data from CSV file:")
    #         for row in nested_data:
    #             print(row)
    #         Mean_data = first_items = [sublist[0] for sublist in nested_data] 
    #         print(Mean_data)

    #         ## make_visualization(nested_data)//untuk 7 titik
    #         # y_coordinates =  [3] * len(Mean_data) 

    #         ## Create X-coordinates for the data points (0 to 6)
    #         # x_coordinates = list(range(len(Mean_data)))

    #         # Create X-coordinates and Y-Coordinate Use 42 point//untuk 42 titik
    #         num_columns = 7
    #         num_rows = len(Mean_data) // num_columns

    #         # Define x and y coordinates for the 42 data points in a grid
    #         x_coordinates = [i % num_columns for i in range(len(Mean_data))]
    #         y_coordinates = [i // num_columns for i in range(len(Mean_data))]
            
    #         # Create a heatmap using matplotlib
    #         plt.scatter(x_coordinates, y_coordinates, c=first_items, cmap='YlGnBu', s=200)
    #         plt.colorbar(label='Values')
    #         plt.title('Heatmap of First Items')
    #         plt.xlabel('X-axis')
    #         plt.ylabel('Y-axis')
            
    #         # Annotate the heatmap with Median values//untuk mengetahui nilai yang ada di heatmap
    #         # for i, median_value in enumerate(Median_data):
    #         #     plt.annotate(str(median_value), (x_coordinates[i], y_coordinates[i]), color='black', fontsize=10, ha='center', va='center')

    #         # Show the plot
    #         plt.show()

    #     else:
    #         print("Data not loaded due to an error or no file selected.")

# excel_file = "Pengujian Awal Wifi Tes F3.xlsx"
# sheetName = "WiFI A"
# distance = input("WiFI distance :")
# coloumn = input("coloumn data :")
# csv_file = "data {0} dengan jarak {1}.csv".format(sheetName,distance)
# convert_xlxs_to_CSV(excel_file, sheetName,coloumn, csv_file)
# print(f"The sheet '{sheetName}' has been converted to CSV and saved as '{csv_file}'.")
# df = pd.read_csv(csv_file)
# signal = df['RSSI'].tolist()
# print(df.columns)
# print(df)

# #calculate kalman filter 
# signal_kalman_filter = kalman_filter(signal, A=1, H=1, Q=1.6, R=6)
# kalmanFilterData_label="kalman_filtered_signal"
# # remove "array" in data 
# signal_kalman_filter = [item[0] if isinsta55nce(item, np.ndarray) else item for item in signal_kalman_filter]
# df[kalmanFilterData_label] = signal_kalman_filter
# df.to_csv(csv_file, index=False)
# #gray filtering
# signal_gray_filter = gray_filter(signal,N=8)
# grayFilterData_label="gray_filtered_signal"
# df[grayFilterData_label] = signal_gray_filter
# df.to_csv(csv_file, index=False)
# #Fourier Filter 
# signal_fft_filter = fft_filter(signal, N=10, M=2)
# fftFilterData_label="fourier_filtered_signal"
# df[fftFilterData_label] = signal_gray_filter
# df.to_csv(csv_file, index=False)



# plot_signals([signal,signal_kalman_filter, signal_gray_filter, signal_fft_filter],
#              ["signal", kalmanFilterData_label, grayFilterData_label, fftFilterData_label]) 