import os
import re
from sklearn.model_selection import train_test_split
import utils
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "modifiedData_42Titik/a23"
output_folder = "centering_data_raw"

RP_points = []
means = []
medians = []
max_values = []

# Define a regex pattern to extract x and y from the file name
pattern = re.compile(r"modified \((\d+),(\d+)\)_a23 - \d{2}-\d{2}-\d{4} \d{2}-\d{2}-\d{2}")


for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        match = pattern.match(file_name)
        if match:

            #  Extract the point information from the file name
            x, y = map(int, match.groups())
            RP_points.append((x, y))

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(os.path.join(folder_path, file_name))

            # Select the specified columns
            selected_columns = ["Wifi_A", "Wifi_A_5G", "Wifi_B", "Wifi_B_5G", "Wifi_C", "Wifi_C_5G", "Wifi_D", "Wifi_D_5G"]
            
            # Calculate mean, median, and max for each selected column
            means.append(df[selected_columns].mean())
            medians.append(df[selected_columns].median())
            max_values.append(df[selected_columns].max())

# Create DataFrames for the results
points_df = pd.DataFrame(RP_points, columns=["x", "y"])
means_df = pd.DataFrame(means, columns=selected_columns)
medians_df = pd.DataFrame(medians, columns=selected_columns)
max_df = pd.DataFrame(max_values, columns=selected_columns)


result_mean = pd.concat([points_df, means_df], axis=1)
result_median = pd.concat([points_df, medians_df], axis=1)
result_max = pd.concat([points_df, max_df], axis=1)


# Save the results to CSV files in the output folder
result_mean.to_csv(os.path.join(output_folder, "mean_results.csv"), index=False)
result_median.to_csv(os.path.join(output_folder, "median_results.csv"), index=False)
result_max.to_csv(os.path.join(output_folder, "max_results.csv"), index=False)


# Create a map of the room
room_length = 7
room_width = 8
room_map = sns.heatmap(data=pd.DataFrame(), annot=True, fmt=".2f", cmap="coolwarm",
                       vmin=0, vmax=100, cbar_kws={'label': 'Signal Strength'})

# Plot mean values
scatter_mean_wifi_A = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Mean Signal Strength for Wifi_A")
plt.colorbar(scatter_mean_wifi_A, label='Signal Strength')
plt.show()

# Plot median values
scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Median Signal Strength for Wifi_A")
plt.colorbar(scatter_median, label='Signal Strength')
plt.show()

# Plot max values
scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Max Signal Strength for Wifi_A")
plt.colorbar(scatter_max, label='Signal Strength')
plt.show()

###########################
scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Mean Signal Strength for Wifi_B")
plt.colorbar(scatter_mean, label='Signal Strength')
plt.show()

# Plot median values
scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Median Signal Strength for Wifi_B")
plt.colorbar(scatter_median, label='Signal Strength')
plt.show()

# Plot max values
scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Max Signal Strength for Wifi_B")
plt.colorbar(scatter_max, label='Signal Strength')
plt.show()

############################
scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Mean Signal Strength for Wifi_C")
plt.colorbar(scatter_mean, label='Signal Strength')
plt.show()

# Plot median values
scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Median Signal Strength for Wifi_C")
plt.colorbar(scatter_median, label='Signal Strength')
plt.show()

# Plot max values
scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Max Signal Strength for Wifi_C")
plt.colorbar(scatter_max, label='Signal Strength')
plt.show()

############################
scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Mean Signal Strength for Wifi_D")
plt.colorbar(scatter_mean, label='Signal Strength')
plt.show()

# Plot median values
scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Median Signal Strength for Wifi_D")
plt.colorbar(scatter_median, label='Signal Strength')
plt.show()

# Plot max values
scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
plt.title("Max Signal Strength for Wifi_D")
plt.colorbar(scatter_max, label='Signal Strength')
plt.show()


# def main() :  
#     folder_path = filedialog.askdirectory()
#     if os.path.exists(folder_path):
#         # List all files in the folder
#         file_list = os.listdir(folder_path)

#         # Iterate through the files in the folder
#         for file_name in file_list:
#             if file_name.endswith('.csv'):
#                 file_path = os.path.join(folder_path, file_name)
#                 df = pd.read_csv(file_path, sep=';')

#                 output_path = os.path.join(folder_path, file_path)

#                 modified_df = pd.read_csv(output_path)
#                 print("Modified CSV File Contents:")
#                 print(modified_df)

#                 rawDataDict = "rawData_42titik_new"

#                 # Create the folder if it doesn't exist
#                 if not os.path.exists(rawDataDict):
#                     os.mkdir(rawDataDict)

#                 #Step 6 calculate max, mean, median from csv 
#                 column_to_analyze = ['Wifi_A', 'Wifi_B', 'Wifi_C', 'Wifi_D', 'Wifi_A_5G', 'Wifi_B_5G', 'Wifi_C_5G', 'Wifi_D_5G']

#                 # calculate and print mean, median, max 
#                 for column_name in column_to_analyze:
#                     stats =calculate_statistics(modified_df, column_name)
#                     print(f"Column: {stats['Column']}")
#                     print(f"Mean: {stats['Mean']}")
#                     print(f"Median: {stats['Median']}")
#                     print(f"Maximum: {stats['Maximum']}")
#                     print()

#                     # Extract the statistics and column name
#                     mean_value = stats['Mean']
#                     median_value = stats['Median']
#                     max_value = stats['Maximum']

#                     # Create a new DataFrame for the statistics
#                     stats_df = pd.DataFrame({'Column': [column_name], 'Mean': [mean_value], 'Median': [median_value], 'Maximum': [max_value]})

#                     # Define the output file names
#                     # rawData = os.path.join(rawDataDict, f'Raw_statData_{column_name}.csv')
#                     rawData_folder = choose_output_folder(os.path.basename(output_path), rawDataDict)
#                     print("rawdatafolder :",rawData_folder)
#                     # Check if the user canceled output folder selection
#                     if rawData_folder == "default_folder":
#                         print(f"Output folder for '{output_path}' selection canceled or not matched.")
#                     else:
#                         print(f"Output folder for '{output_path}' selected: {rawData_folder}")
#                         rawData = os.path.join(rawData_folder, f'Raw_statData_{column_name}.csv')
#                         print(rawData)

#                         # Append the statistics to the existing or new CSV file
#                         utils.append_to_csv(rawData, [stats], headers=['Column', 'Mean', 'Median', 'Maximum'])  

# if __name__ == "__main__":
#     main()