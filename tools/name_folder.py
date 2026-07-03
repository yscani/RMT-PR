import os


def create_folders_in_subfolders(folder_path, new_folder_name):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        new_folder_path = os.path.join(subfolder, new_folder_name)
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created new folder: {new_folder_path}")
        else:
            print(f"Folder already exists, skipping: {new_folder_path}")


def rename_folders_in_subfolders(folder_path, old_folder_name, new_folder_name):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for subfolder in subfolders:
        folder_list = [f.name for f in os.scandir(subfolder) if f.is_dir()]
        if old_folder_name in folder_list:
            old_folder_path = os.path.join(subfolder, old_folder_name)
            new_folder_path = os.path.join(subfolder, new_folder_name)
            os.rename(old_folder_path, new_folder_path)
            print(f"Renamed folder from {old_folder_path} to {new_folder_path}")


if __name__ == "__main__":
    folder_path = "/path/to/data_root_folder"
    old_folder_name = "depth_map_50"
    new_folder_name = "depth_map"
    rename_folders_in_subfolders(folder_path, old_folder_name, new_folder_name)
