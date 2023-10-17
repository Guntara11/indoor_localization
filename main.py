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


def filter_data_by_column_name(df, keyword):
    # Filter columns that contain the keyword
    wifi_columns = [col for col in df.columns if keyword in col]
    wifi_data = df[wifi_columns]
    return wifi_data


def partioning_data(df, test_size = 0.2):
    bayesian_data, kalman_data = train_test_split(df, test_size=test_size)
    return bayesian_data.values.tolist(), kalman_data.values.tolist()

def choose_output_folder(input_file_name):
    if "_a23" in input_file_name:
        return os.path.join(rawDataDict, "a23")
    elif "_f3" in input_file_name:
        return os.path.join(rawDataDict, "f3")
    else:
        # Default output folder if no match is found
        return "default_folder"



if __name__ == "__main__":
    # Step 1: Open the CSV file with a specified separator
    df, file_path = open_csv_file(separator=';')  # You can specify the separator you want
    

    if df is not None:
        kalmanFilterData_label="kalman_filtered_signal"
        file_name = os.path.basename(file_path)
        # Step 2: Count the number of columns in the opened CSV file
        num_columns = count_columns(df)

        print("CSV FILE : ",file_name)
        # Step 3 and Step 4: Modify the CSV file as needed
        output_path = "newData/modified {0}".format(file_name)
        modify_csv(df, separator=',', output_path= output_path)  # Change the separator if needed

        # Step 5: Open the modified CSV using Pandas and print its content
        modified_df = pd.read_csv(output_path)
        print("Modified CSV File Contents:")
        print(modified_df)

        rawDataDict = "newRawData"

        # Create the folder if it doesn't exist
        if not os.path.exists(rawDataDict):
            os.mkdir(rawDataDict)
        #Step 6 calculate max, mean, median from csv 
        column_to_analyze = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D', 'Wifi_A_5G', 'Wifi_B_5G', 'Wifi_C_5G', 'Wifi_D_5G']

        # calculate and print mean, median, max 
        for column_name in column_to_analyze:
            stats =calculate_statistics(modified_df, column_name)
            print(f"Column: {stats['Column']}")
            print(f"Mean: {stats['Mean']}")
            print(f"Median: {stats['Median']}")
            print(f"Maximum: {stats['Maximum']}")
            print()
        
            # Extract the statistics and column name
            mean_value = stats['Mean']
            median_value = stats['Median']
            max_value = stats['Maximum']

            # Create a new DataFrame for the statistics
            stats_df = pd.DataFrame({'Column': [column_name], 'Mean': [mean_value], 'Median': [median_value], 'Maximum': [max_value]})

            # Define the output file names
            # rawData = os.path.join(rawDataDict, f'Raw_statData_{column_name}.csv')
            rawData_folder = choose_output_folder(os.path.basename(output_path))
            print("rawdatafolder :",rawData_folder)
            # Check if the user canceled output folder selection
            if rawData_folder == "default_folder":
                print(f"Output folder for '{output_path}' selection canceled or not matched.")
            else:
                print(f"Output folder for '{output_path}' selected: {rawData_folder}")
                rawData = os.path.join(rawData_folder, f'Raw_statData_{column_name}.csv')
                print(rawData)

                # Append the statistics to the existing or new CSV file
                utils.append_to_csv(rawData, [stats], headers=['Column', 'Mean', 'Median', 'Maximum'])  




            # Define the output file name
            # output_Raw_file_name ="newRawData/RAW_{0}.csv".format(column_name)
            # Save the statistics DataFrame to a new CSV file with the appropriate headers
            # stats_df.to_csv(output_Raw_file_name, index=False)
       
        
        # Step 6 
        # rawDataDict = "newRawData"
        # rawData = {}
        # for column_name in modified_df:
        #     # Create a new DataFrame with just the selected column
        #     new_df = modified_df[[column_name]]
        #     # Define the output file 
        #     outputRawData = "newRawData/RAW_{0}.csv".format(column_name)
        #     # save data 
        #     # new_df.to_csv(outputRawData, index=False)
        #     #     # Check if the output file already exists
        #     try:
        #     # Append the new data to the existing file
        #         # Try to load the existing output file (if it exists)
        #         existing_df = pd.read_csv(outputRawData)
        #         # Append the new data to the existing file
        #         combined_df = pd.concat([existing_df, new_df], axis=1)
        #         # Save the updated data without headers
        #         combined_df.to_csv(outputRawData, index=False)
        #     except FileNotFoundError:
        #         # If the file doesn't exist, create a new one
        #         new_df.to_csv(outputRawData, index=False)
        # for rawFileName in os.listdir(rawDataDict):
        #     if rawFileName.endswith('.csv'):
        #         column_name = os.path.splitext(rawFileName)[0]
        #         raw_df = pd.read_csv(os.path.join(rawDataDict, rawFileName))
        #         data_list = raw_df.values.ravel().tolist()
        #         rawData[column_name] = data_list
        # for column_name, data_list in rawData.items():
        #     print(f'Data for {column_name}: {data_list}')


        # selected_data = filter_data_by_column_name(modified_df, keyword="Wifi")
        # selected_columns = input("Enter the column names (comma-separated) to selet: ").split(',')
        # selected_data = modified_df.loc[:, selected_columns]


# partioning data 

        # bayesian_data, kalman_data = partioning_data(selected_data)

        # flat_bayesian_data = [item for sublist in bayesian_data for item in sublist]
        # flat_kalman_data = [item for sublist in kalman_data for item in sublist]

        # print("Bayesian Data (80%):", flat_bayesian_data )
        # print("kalman Data (20%):", flat_kalman_data )

        # signal_kalman_filter = kalman_filter(kalman_data, A=1, H=1, Q=1.6, R=6)
        # signal_kalman_filter = [item[0] if isinstance(item, np.ndarray) else item for item in signal_kalman_filter]
        # flat_kalman_signal = [item for sublist in signal_kalman_filter for item in (sublist if isinstance(sublist, list) else [sublist])]
        # print("filtered data:", flat_kalman_data)



        # # Create a grid of x and y coordinates that match the data
        # x_coordinates = np.arange(len(bayesian_data))
        # y_coordinates = np.array([1])


        # data_array = np.array([bayesian_data])

        # # Create the heatmap
        # plt.figure(figsize=(8, 2))  # Adjust the figsize as needed
        # plt.imshow(data_array, cmap='viridis', origin='lower', aspect='auto')

        # # Add labels and colorbar
        # plt.xlabel('Time Step')
        # plt.ylabel('Row')
        # plt.title('Wi-Fi RSSI Heatmap')
        # plt.colorbar(label='RSSI (dBm)')

        # # Customize x and y axis labels based on your data
        # plt.xticks(np.arange(len(x_coordinates)), x_coordinates)
        # plt.yticks(np.arange(len(y_coordinates)), y_coordinates)

        # # Show the plot
        # plt.show()
        # plot_signals([kalman_data, signal_kalman_filter], ["signal", kalmanFilterData_label])

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
            






