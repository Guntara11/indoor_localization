import os
import re
from sklearn.model_selection import train_test_split
import utils
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "filtered_calibrated_dataset/test/a23/partition_5"
output_folder = "centering_data_raw"

RP_points = []
means = []
medians = []
max_values = []

# Define a regex pattern to extract x and y from the file name
# pattern = re.compile(r"modified \((\d+),(\d+)\)_a23 - \d{2}-\d{2}-\d{4} \d{2}-\d{2}-\d{2}")
pattern = re.compile(r"test_modified_\((\d+),(\d+)\)_a23 - \d{2}-\d{2}-\d{4} \d{2}-\d{2}-\d{2}")


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
# round the value because to many number behind the separator
means_df = means_df.round(2)

result_mean = pd.concat([points_df, means_df], axis=1)
result_median = pd.concat([points_df, medians_df], axis=1)
result_max = pd.concat([points_df, max_df], axis=1)


# Save the results to CSV files in the output folder
result_mean.to_csv(os.path.join(output_folder, "mean_results_skema4_part5.csv"), index=False)
result_median.to_csv(os.path.join(output_folder, "median_results_skema4_part5.csv"), index=False)
result_max.to_csv(os.path.join(output_folder, "max_results_skema4_part5.csv"), index=False)


############################################## section for plotting ###########################################################
fig_mean, axes_mean = plt.subplots(1, len(selected_columns), figsize=(16, 4), sharex=True, sharey=True)

for i, wifi_type in enumerate(selected_columns):
    axes_mean[i].scatter(points_df["y"], points_df["x"], c=means_df[wifi_type], cmap='YlGnBu', s=100, marker='s')
    axes_mean[i].set_aspect('equal', adjustable='box')
    axes_mean[i].set_title(f'Mean\n{wifi_type}')
    axes_mean[i].set_xlabel('X-axis')
    axes_mean[i].set_ylabel('Y-axis')

# Create the colorbar and specify the axes to use
cbar_ax_mean = fig_mean.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar_mean = fig_mean.colorbar(axes_mean[0].collections[0], cax=cbar_ax_mean, label='Values')
fig_mean.suptitle("Mean Signal Strength for Each WiFi Type")
fig_mean.savefig("fingerprint/mean_signal_strength_skema4_part5.png")
plt.show()

fig_median, axes_median = plt.subplots(1, len(selected_columns), figsize=(16, 4), sharex=True, sharey=True)

for i, wifi_type in enumerate(selected_columns):
    axes_median[i].scatter(points_df["y"], points_df["x"], c=medians_df[wifi_type], cmap='YlGnBu', s=100, marker='s')
    axes_median[i].set_aspect('equal', adjustable='box')
    axes_median[i].set_title(f'Median\n{wifi_type}')
    axes_median[i].set_xlabel('X-axis')
    axes_median[i].set_ylabel('Y-axis')

# Create the colorbar and specify the axes to use
cbar_ax_median = fig_median.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar_median = fig_median.colorbar(axes_median[0].collections[0], cax=cbar_ax_median, label='Values')
fig_median.suptitle("Median Signal Strength for Each WiFi Type")
fig_mean.savefig("fingerprint/median_signal_strength_skema4_part5.png")
plt.show()

# Plot max for each WiFi type
fig_max, axes_max = plt.subplots(1, len(selected_columns), figsize=(16, 4), sharex=True, sharey=True)

for i, wifi_type in enumerate(selected_columns):
    axes_max[i].scatter(points_df["y"], points_df["x"], c=max_df[wifi_type], cmap='YlGnBu', s=100, marker='s')
    axes_max[i].set_aspect('equal', adjustable='box')
    axes_max[i].set_title(f'Max\n{wifi_type}')
    axes_max[i].set_xlabel('X-axis')
    axes_max[i].set_ylabel('Y-axis')

# Create the colorbar and specify the axes to use
cbar_ax_max = fig_max.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar_max = fig_max.colorbar(axes_max[0].collections[0], cax=cbar_ax_max, label='Values')
fig_max.suptitle("Max Signal Strength for Each WiFi Type")
fig_mean.savefig("fingerprint/max_signal_strength_skema4_part5.png")
plt.show()
