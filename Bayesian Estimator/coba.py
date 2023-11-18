import os
import numpy as np

# Replace 'Raw/Alba' with the actual path to your directory
directory_path = 'Raw/Alba'

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)

        # Read CSV file
        csv_data = np.genfromtxt(file_path, delimiter=',')

        # Extract columns 5 to 205 for each row
        selected_data = csv_data[:, 5:206]

        # Calculate mean, median, and maximum for each row
        mean_values = np.mean(selected_data, axis=1)
        median_values = np.median(selected_data, axis=1)
        max_values = np.max(selected_data, axis=1)

        # Print or use the calculated values as needed
        for i in range(len(mean_values)):
            print(f"File: {filename}, Row {i + 1}: Mean = {mean_values[i]}, Median = {median_values[i]}, Max = {max_values[i]}")