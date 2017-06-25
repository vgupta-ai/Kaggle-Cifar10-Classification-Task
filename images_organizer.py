import os.path
from shutil import copyfile


kaggle_train_labels_file = "trainLabels.csv"

#def copy_file_to_location(file_source_path,file_destination_path):


def arrange_files_into_classes_from_directory(directory_name,filename_label_map):
    with open(filename_label_map) as f:
        all_rows = f.readlines()
        is_first_row = True
        for row in all_rows:
            row = row.strip()
            if is_first_row:
                is_first_row = False
                continue
            else:
                file_name,file_class_name = row.split(",")
                file_name = file_name+".png"
                file_source_path = os.path.join(directory_name,file_name)
                destination_directory_path = os.path.join(directory_name+"_organized",file_class_name)
                if not os.path.exists(destination_directory_path):
                    os.makedirs(destination_directory_path)
                file_destination_path = os.path.join(destination_directory_path,file_name)
                copyfile(file_source_path, file_destination_path)

arrange_files_into_classes_from_directory("train",kaggle_train_labels_file)
