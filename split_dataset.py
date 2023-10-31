import os
import shutil

#memindahkan ke folder partition
def partition (dataset_split_path, dataset_file, dataset_file_origin):
    if "partition_1" in dataset_file:
        dataset_folder_destination_satu = os.path.join(dataset_split_path, "partition_1")
        dataset_file_destination_satu = os.path.join(dataset_folder_destination_satu, dataset_file)

        if "partition_10" in dataset_file:
            dataset_folder_destination = os.path.join(dataset_split_path, "partition_10")
            dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

            shutil.move(dataset_file_origin, dataset_file_destination)
        else:
            shutil.move(dataset_file_origin, dataset_file_destination_satu)

    elif "partition_2" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_2")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_3" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_3")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_4" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_4")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_5" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_5")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_6" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_6")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_7" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_7")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_8" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_8")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)
        
        shutil.move(dataset_file_origin, dataset_file_destination)

    elif "partition_9" in dataset_file:
        dataset_folder_destination = os.path.join(dataset_split_path, "partition_9")
        dataset_file_destination = os.path.join(dataset_folder_destination, dataset_file)

        shutil.move(dataset_file_origin, dataset_file_destination)

#folder path dataset
dataset_path = "dataset"

#folder path dataset test dan train
dataset_test_path = os.path.join(dataset_path, "test")
dataset_train_path = os.path.join(dataset_path, "train")

#folder path dataset test a23 dan f3
dataset_test_a23_path = os.path.join(dataset_test_path, "a23")
dataset_test_f3_path = os.path.join(dataset_test_path, "f3")

#folder path dataset train a23 dan f3
dataset_train_a23_path = os.path.join(dataset_train_path, "a23")
dataset_train_f3_path = os.path.join(dataset_train_path, "f3")

if os.path.exists(dataset_path):
    dataset_list_path = os.listdir(dataset_path)

    #iterasi list file dataset
    for dataset_file in dataset_list_path:
        
        #filter dataset test a23 (partisi 1 - 10)
        if "test_modified" in dataset_file and "_a23" in dataset_file:
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            partition(dataset_split_path=dataset_test_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset test f3 (partisi 1 - 10)
        elif ("test_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_test_f3_path, dataset_file)
            partition(dataset_split_path=dataset_test_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset train a23 (partisi 1 - 10)
        elif "train_modified" in dataset_file and "_a23" in dataset_file:
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_train_a23_path, dataset_file)
            partition(dataset_split_path=dataset_train_a23_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)

        #filter dataset train f3 (partisi 1 - 10)
        elif ("train_modified" in dataset_file) and ("_f3" in dataset_file or "_F3" in dataset_file or "_f23" in dataset_file):
            dataset_file_origin = os.path.join(dataset_path, dataset_file)
            dataset_file_destination = os.path.join(dataset_train_f3_path, dataset_file)
            partition(dataset_split_path=dataset_train_f3_path, dataset_file=dataset_file, dataset_file_origin=dataset_file_origin)
