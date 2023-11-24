import os
import shutil

def rename_files(folder_path, device_name, data_type):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if data_type in file:
                old_path = os.path.join(root, file)

                # Extracting coordinates (x, y) from the old file name
                coordinates = file.split('_')[2]
                
                # Creating the new file name
                new_file_name = f"{data_type}_{coordinates}_{device_name}"

                # Creating the new file path
                new_path = os.path.join(root, new_file_name)

                # Renaming the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} to {new_path}")

# Example usage
# rename_files("dataset/train/a23", "a23", "train")
# rename_files("dataset/test/a23", "a23", "test")
rename_files("dataset/train/f3", "f3", "train")
rename_files("dataset/test/f3", "f3", "test")