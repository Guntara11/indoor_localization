import os 
import pandas as pd
import math
import matplotlib.pyplot as plt 
import numpy as np
import tkinter as tk 
from tkinter import filedialog
import csv 
from scipy.spatial import distance
"""
'''
Author: Ega Guntara
Date: 23/09/2023
utility file that is the main backprocesss for 
main code 
'''  
"""

def open_csv_file(separator=','):
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if file_path:
        try:
            # Read the CSV file with the specified separator
            df = pd.read_csv(file_path, sep=separator)
            nested_data = df[['Mean', 'Median','Maximum']].values.tolist()
            return nested_data, file_path
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {str(e)}")
            return None, None
    else:
        print("No file selected.")
        return None, None

def modify_csv(df, separator=',', output_path = None):
    if df is not None:
        num_columns = len(df.columns)
        print(f"Number of columns in the CSV file: {num_columns}")

        if num_columns == 1:
            # Split the single column into multiple columns using the separator
            df = df[df.columns[0]].str.split(separator, expand=True)
            print("CSV File Contents (after splitting):")
            print(df)
                # If an output path is provided, save the modified CSV there
        
        if output_path:
            df.to_csv(output_path, sep=',', index=False)
            print(f"Modified CSV file saved at: {output_path}")
        else:
            # Change the separator to ","
            df.to_csv("modified_csv.csv", sep=',', index=False)
            print("CSV file modified and saved as 'modified_csv.csv'.")
        # Change the separator to ","
        df.to_csv("modified_csv.csv", sep=',', index=False)
        print("CSV file modified and saved as 'modified_csv.csv'.")
    else:
        print("CSV file could not be opened.")

def count_columns(df):
    if df is not None:
        return len(df.columns)
    else:
        print("DataFrame is None; unable to count columns.")
        return None
# def calculate_statistics(df, column_name):
#     # Load your output CSV file
#     column_data = df[column_name]
#     mean_value = column_data.mean()
#     median_value = column_data.median()
#     max_value = column_data.max()
    
#     return {
#         'Column': column_name,
#         'Mean': mean_value,
#         'Median': median_value,
#         'Maximum': max_value
#     }

#     # Calculate mean, median, and maximum for each column
#     results = {
#         'WIFI_A_mean': df['WIFI_A'].mean(),
#         'WIFI_B_mean': df['WIFI_B'].mean(),
#         'WIFI_C_mean': df['WIFI_C'].mean(),
#         'WIFI_D_mean': df['WIFI_D'].mean(),
#         'WIFI_A_5G_mean': df['WIFI_A_5G'].mean(),
#         'WIFI_B_5G_mean': df['WIFI_B_5G'].mean(),
#         'WIFI_C_5G_mean': df['WIFI_C_5G'].mean(),
#         'WIFI_D_5G_mean': df['WIFI_D_5G'].mean(),
#         'WIFI_A_median': df['WIFI_A'].median(),
#         'WIFI_B_median': df['WIFI_B'].median(),
#         'WIFI_C_median': df['WIFI_C'].median(),
#         'WIFI_D_median': df['WIFI_D'].median(),
#         'WIFI_A_5G_median': df['WIFI_A_5G'].median(),
#         'WIFI_B_5G_median': df['WIFI_B_5G'].median(),
#         'WIFI_C_5G_median': df['WIFI_C_5G'].median(),
#         'WIFI_D_5G_median': df['WIFI_D_5G'].median(),
#         'WIFI_A_max': df['WIFI_A'].max(),
#         'WIFI_B_max': df['WIFI_B'].max(),
#         'WIFI_C_max': df['WIFI_C'].max(),
#         'WIFI_D_max': df['WIFI_D'].max(),
#         'WIFI_A_5G_max': df['WIFI_A_5G'].max(),
#         'WIFI_B_5G_max': df['WIFI_B_5G'].max(),
#         'WIFI_C_5G_max': df['WIFI_C_5G'].max(),
#         'WIFI_D_5G_max': df['WIFI_D_5G'].max(),
#     }
#     return results
def calculate_metrics(folder_path):
    # Mengambil daftar nama file dalam folder
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Inisialisasi list untuk menyimpan hasil dari setiap file
    mean_list = []
    median_list = []
    max_list = []

    # Iterasi melalui setiap file
    for file_name in file_names:
        # Membaca file CSV
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # Memilih kolom Wifi_A sampai Wifi_D
        wifi_columns = df[['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D']]

        # Menghitung nilai mean, median, dan max untuk setiap kolom
        mean_values = np.mean(wifi_columns, axis=0)
        median_values = np.median(wifi_columns, axis=0)
        max_values = np.max(wifi_columns, axis=0)

        # Menambahkan hasil ke dalam list
        mean_list.append(mean_values)
        median_list.append(median_values)
        max_list.append(max_values)

    # Mengonversi hasil ke dalam ndarray
    mean_array = np.array(mean_list)
    median_array = np.array(median_list)
    max_array = np.array(max_list)

    return mean_array, median_array, max_array

def euclidean_distance(point1, point2):
    return distance.euclidean(point1, point2)
    # Convert the results dictionary to a DataFrame
    # results_df = pd.DataFrame([results])

    # Save the results to a new CSV file
    # results_df.to_csv(output_file, index=False) 
def append_to_csv(file_path, data, headers=None):
    # Append data to an existing CSV file or create a new one
    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path)
    else:
        existing_df = pd.DataFrame(columns=headers)
    
    new_data = pd.DataFrame(data, columns=headers)
    updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    updated_df.to_csv(file_path, index=False)
def convert_xlxs_to_CSV(excel_file_path, sheetName, col, csv_file_path):
    """
    taking data in excel coloumn F sheet WiFI A
    and take only data without other useless data like string or other
    the data RSSI is being labeled as "RSSI" to make easy take the data to csv 
    the data is being converted to csv file because it much easier to extract the 
    data if the data file is .csv  
    """
    excel_file_path = os.path.abspath(excel_file_path)
    csv_file_path = os.path.abspath(csv_file_path)
    df = pd.read_excel(excel_file_path, sheet_name= sheetName, usecols=col)
    selectedData= df.iloc[5:44]
    if 'Unnamed: 5' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 5': 'RSSI'}, inplace=True)
    if 'Unnamed: 6' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 6': 'RSSI'}, inplace=True)
    if 'Unnamed: 7' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 7': 'RSSI'}, inplace=True)
    if 'Unnamed: 8' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 8': 'RSSI'}, inplace=True)
    if 'Unnamed: 9' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 9': 'RSSI'}, inplace=True)
    if 'Unnamed: 10' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 10': 'RSSI'}, inplace=True)
    if 'Unnamed: 11' in selectedData.columns:
        selectedData.rename(columns={'Unnamed: 11': 'RSSI'}, inplace=True)   
    selectedData.insert(0, 'ID', range(1, len(selectedData) + 1))
    selectedData.to_csv(csv_file_path, index= False)

def choose_output_folder(input_file_name, rawDataDict):
        if "_a23" in input_file_name:
            return os.path.join(rawDataDict, "a23")
        elif "_f3" or "F3" or "f23" in input_file_name:
            return os.path.join(rawDataDict, "f3")
        else:
            # Default output folder if no match is found
            return "default_folder"


def plot_signals(signals, labels):

    """

    Auxiliary function to plot all signals.

    input:
        - signals: signals to plot
        - labels: labels of input signals

    output:
        - display plot

    """

    alphas = [1, 0.45, 0.45, 0.45, 0.45]      # just some opacity values to facilitate visualization

    lenght = np.shape(signals)[1]             # time lenght of original and filtered signals

    plt.figure()

    for j, sig in enumerate(signals):          # iterates on all signals
        if isinstance(sig, np.ndarray):  # Check if it's a NumPy array
            sig = sig[0]  # Convert to a numeric value if it's an array
        plt.plot(range(lenght), sig, '-o', label=labels[j], markersize=2, alpha=alphas[j])

    plt.grid()

    plt.ylabel('RSSI')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    return
def bayesian_likelihood_function(distance, standard_deviation):
    return math.exp(-(distance ** 2) / (2 * standard_deviation ** 2))

def hist_gauss(values):
    val = sorted(values[values!=-100.0])
    x = range(-100, -10)
    mu = np.mean(val)
    sigma = np.std(val)
    if np.isnan(mu) and np.isnan(sigma):
        y = [0]*len(x)
    elif mu==0 or sigma==0:
        y = [0]*len(x)
    else:
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
        (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))
    return y

def kalman_block(x, P, s, A, H, Q, R):

    """
    Prediction and update in Kalman filter

    input:
        - signal: signal to be filtered
        - x: previous mean state
        - P: previous variance state
        - s: current observation
        - A, H, Q, R: kalman filter parameters

    output:
        - x: mean state prediction
        - P: variance state prediction

    """

    # check laaraiedh2209 for further understand these equations

    x_mean = A * x + np.random.normal(0, Q, 1)
    P_mean = A * P * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    x = x_mean + K * (s - H * x_mean)
    P = (1 - K * H) * P_mean

    return x, P

def kalman_filter(signal, A, H, Q, R):

    """

    Implementation of Kalman filter.
    Takes a signal and filter parameters and returns the filtered signal.

    input:
        - signal: signal to be filtered
        - A, H, Q, R: kalman filter parameters

    output:
        - filtered signal

    """

    predicted_signal = []

    x = signal[0]                                 # takes first value as first filter prediction
    P = 0                                         # set first covariance state value to zero

    predicted_signal.append(x)
    for j, s in enumerate(signal[1:]):            # iterates on the entire signal, except the first element

        x, P = kalman_block(x, P, s, A, H, Q, R)  # calculates next state prediction

        predicted_signal.append(x)                # update predicted signal with this step calculation

    return predicted_signal


def gray_filter(signal, N=15):

    """
    Implementation of Gray filter.
    Takes a signal and filter parameters and return the filtered signal.

    input:
        - signal: signal to be filtered
        - N: window size of input signal

    output:
        - filtered signal
    """

    predicted_signal = []

    for j in range(0, np.shape(signal)[0], N):    # iterates on the entire signal, taking steps by N (window size)

        N = np.minimum(N, np.shape(signal)[0]-j)  # just in case we are at signal final and N samples are not available

        R_0 = np.zeros(N)
        R_0[:] = signal[j:j+N]                     # saves in R_0 signal values of corresponding window size

        R_1 = []

        for i in range(N):

            R_1.append((np.cumsum(R_0[0:i+1]))[i])  # calculates R_1

        # calculates gray filter solution

        # for further details about filter resolution check kayacan2010

        B = (np.matrix([np.ones((N-1)), np.ones((N-1))])).T

        for k in range(N-1):

            B[k, 0] = -0.5 * (R_1[k+1] + R_1[k])

        X_n = np.matrix(np.asarray(R_0[1:])).T

        _ = np.matmul(np.linalg.inv(np.matmul(B.T, B)), (np.matmul(B.T, X_n)))

        a = _[0, 0]
        u = _[1, 0]

        X_ = R_0[0]
        predicted_signal.append(X_)
        for i in range(1, N):                                  # update predicted signal with this window calculation
            predicted_signal.append((((R_0[0] - u/a) * np.exp(-a * (i - 1)))*(1 - np.exp(a))))

    return predicted_signal
def fft_filter(signal, N=8, M=2):

    """
    Implementation of Fourier filter.
    Takes a signal and filter parameters and return the filtered signal.

    input:
        - signal: signal to be filtered
        - N: window size of input signal
        - M: samples of fft signal to preserve (remember fft symmetry)

    output:
        - filtered signal
    """

    predicted_signal = []

    for j in range(0, np.shape(signal)[0], N):      # iterates on the entire signal, taking steps by N (window size)

        N = np.minimum(N, np.shape(signal)[0] - j)  # just in case we are at signal final and N samples are not avail

        R_0 = np.zeros(N)
        R_0[:] = signal[j:j+N]                       # saves in R_0 signal values of corresponding window size

        R_0_fft = np.fft.fft(R_0)                    # fft of signal window

        for k in range(int(N / 2)):                    # it keeps M samples of fft and sets the rest to zero
            R_0_fft[M+k] = 0                         # remember fft symmetry
            R_0_fft[-1-M-k] = 0

        R_0_ifft = np.fft.ifft(R_0_fft)              # inverse fft

        for i in range(0, N):
            predicted_signal.append(R_0_ifft[i])     # update predicted signal with this window calculation

    return predicted_signal