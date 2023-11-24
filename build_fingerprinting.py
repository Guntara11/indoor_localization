import os
import re
from sklearn.model_selection import train_test_split
import utils
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "dataset/test/a23/partition_1"
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
result_mean.to_csv(os.path.join(output_folder, "mean_results.csv"), index=False)
result_median.to_csv(os.path.join(output_folder, "median_results.csv"), index=False)
result_max.to_csv(os.path.join(output_folder, "max_results.csv"), index=False)


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
fig_mean.savefig("fingerprint/mean_signal_strength.png")
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
fig_mean.savefig("fingerprint/median_signal_strength.png")
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
fig_mean.savefig("fingerprint/max_signal_strength.png")
plt.show()

#########################uncomment this if u want to make of room an RP points with centering data as data in RP points################################
# Create a map of the room
# room_length = 7
# room_width = 8
# room_map = sns.heatmap(data=pd.DataFrame(), annot=True, fmt=".2f", cmap="coolwarm",
#                        vmin=0, vmax=100, cbar_kws={'label': 'Signal Strength'})

# # Plot mean values
# scatter_mean_wifi_A = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Mean Signal Strength for Wifi_A")
# plt.colorbar(scatter_mean_wifi_A, label='Signal Strength')
# plt.show()

# # Plot median values
# scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Median Signal Strength for Wifi_A")
# plt.colorbar(scatter_median, label='Signal Strength')
# plt.show()

# # Plot max values
# scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_A"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Max Signal Strength for Wifi_A")
# plt.colorbar(scatter_max, label='Signal Strength')
# plt.show()

# ###########################
# scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Mean Signal Strength for Wifi_B")
# plt.colorbar(scatter_mean, label='Signal Strength')
# plt.show()

# # Plot median values
# scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Median Signal Strength for Wifi_B")
# plt.colorbar(scatter_median, label='Signal Strength')
# plt.show()

# # Plot max values
# scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_B"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Max Signal Strength for Wifi_B")
# plt.colorbar(scatter_max, label='Signal Strength')
# plt.show()

# ############################
# scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Mean Signal Strength for Wifi_C")
# plt.colorbar(scatter_mean, label='Signal Strength')
# plt.show()

# # Plot median values
# scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Median Signal Strength for Wifi_C")
# plt.colorbar(scatter_median, label='Signal Strength')
# plt.show()

# # Plot max values
# scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_C"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Max Signal Strength for Wifi_C")
# plt.colorbar(scatter_max, label='Signal Strength')
# plt.show()

# ############################
# scatter_mean = plt.scatter(x=points_df["y"], y=points_df["x"], c=means_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Mean Signal Strength for Wifi_D")
# plt.colorbar(scatter_mean, label='Signal Strength')
# plt.show()

# # Plot median values
# scatter_median = plt.scatter(x=points_df["y"], y=points_df["x"], c=medians_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Median Signal Strength for Wifi_D")
# plt.colorbar(scatter_median, label='Signal Strength')
# plt.show()

# # Plot max values
# scatter_max = plt.scatter(x=points_df["y"], y=points_df["x"], c=max_df["Wifi_D"], marker="o", s=100, edgecolor='w', linewidth=0.5, cmap="coolwarm")
# plt.title("Max Signal Strength for Wifi_D")
# plt.colorbar(scatter_max, label='Signal Strength')
# plt.show()
