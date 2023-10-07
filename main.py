import os
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


def select_data(df, coloumns):
    return df[coloumns]


def partioning_data(df, test_size = 0.2):
    bayesian_data, kalman_data = train_test_split(df, test_size=test_size)
    return bayesian_data.values.tolist(), kalman_data.values.tolist()

def KalmanProcess(signal, A=1, H=1, Q=1.6, R=6):
    signal_kalman_filter = kalman_filter(signal, A=1, H=1, Q=1.6, R=6)
    kalmanFilterData_label="kalman_filtered_signal"

if __name__ == "__main__":
    # Step 1: Open the CSV file with a specified separator
    df, file_path = open_csv_file(separator=';')  # You can specify the separator you want

    if df is not None:
        file_name = os.path.basename(file_path)
        # Step 2: Count the number of columns in the opened CSV file
        num_columns = count_columns(df)

        print("CSV FILE : ",file_name)
        # Step 3 and Step 4: Modify the CSV file as needed
        output_path = "newData/modified {0}.csv".format(file_name)
        modify_csv(df, separator=',', output_path= output_path)  # Change the separator if needed

        # Step 5: Open the modified CSV using Pandas and print its content
        modified_df = pd.read_csv(output_path)
        print("Modified CSV File Contents:")
        print(modified_df)


        selected_columns = input("Enter the column names (comma-separated) to select: ").split(',')
        selected_data = modified_df.loc[:, selected_columns]

        bayesian_data, kalman_data = partioning_data(selected_data)

        print("Bayesian Data (80%):", bayesian_data )
        print("kalman Data (20%):", kalman_data )
    else:
        print("CSV file could not be opened.")
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
            






