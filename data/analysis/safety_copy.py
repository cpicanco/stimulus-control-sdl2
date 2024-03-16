from fileutils import cd, list_data_folders, list_files, safety_copy

def make_safety_copy():
    cd('..')

    data_folder = list_data_folders()

    for folder_name in data_folder:
        cd(folder_name)
        for entry in list_files(''):
            safety_copy(entry)
        cd('..')

if __name__ == "__main__":
    make_safety_copy()