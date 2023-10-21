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
# data = [-50.55, -45.7, -40.8, -46.6, -46.25, -43.15, -46.1]
# coordinates = [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]

# # Create a 2D grid for the heatmap
# heatmap = np.zeros((7, 4))  # Assuming 7 rows and 4 columns

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

# if __name__ == "__main__":
#     nested_data, file_path = open_csv_file(separator=',')
    
#     if nested_data:
#         # Process and work with the nested_data as needed in your main script
#         print("Data from CSV file:")
#         for row in nested_data:
#             print(row)
#         print(nested_data)
#         make_visualization(nested_data)


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
            






