import os
from sklearn.model_selection import train_test_split
import utils
from utils import *

def choose_output_folder(input_file_name, rawDataDict):
        if "_a23" in input_file_name:
            return os.path.join(rawDataDict, "a23")
        elif "_f3" in input_file_name:
            return os.path.join(rawDataDict, "f3")
        else:
            # Default output folder if no match is found
            return "default_folder"


def main() :  
    folder_path = filedialog.askdirectory()
    if os.path.exists(folder_path):
        # List all files in the folder
        file_list = os.listdir(folder_path)

        # Iterate through the files in the folder
        for file_name in file_list:
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, sep=';')

                output_path = os.path.join(folder_path, file_path)

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
                    rawData_folder = choose_output_folder(os.path.basename(output_path), rawDataDict)
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

if __name__ == "__main__":
    main()
