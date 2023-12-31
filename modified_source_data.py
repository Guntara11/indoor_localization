import os
import pandas as pd
import utils
from utils import *


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    folder_path = filedialog.askdirectory()
    if os.path.exists(folder_path):
        # List all files in the folder
        file_list = os.listdir(folder_path)

        # Iterate through the files in the folder
        for file_name in file_list:
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, sep=';')
                output_path_a23 = "modified_skema2/a23/modified_{0}".format(file_name)
                output_path_f3 = "modified_skema2/f3/modified_{0}".format(file_name)
                if ("_f3" in file_path or "_f23" in file_path or "_F3" in file_path):
                    modify_csv(df, separator=';', output_path= output_path_f3)
                elif("_a23" in file_path):
                    modify_csv(df, separator=',', output_path= output_path_a23)
                else:
                    print("wrong output directory")
                
                # file_name = os.path.basename(file_path)

                # num_columns = count_columns(df)

                # print("CSV FILE : ",file_name)
                # Step 3 and Step 4: Modify the CSV file as needed
                # modify_csv(df, separator=',', output_path= output_path)  # Change the separator if needed

                # Step 5: Open the modified CSV using Pandas and print its content
                # modified_df = pd.read_csv(output_path)
                # print("Modified CSV File Contents:")
                # print(modified_df)
if __name__ == "__main__":
    main()