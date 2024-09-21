from fileutils import cd, list_data_folders, list_files, safety_copy, get_cycle

def make_safety_copy():
    cd('..')

    data_folder = list_data_folders()

    for folder_name in data_folder:
        cd(folder_name)
        last_entry = ''
        for entry in list_files(''):
            entry_name = entry.split('.')[0]
            if last_entry != entry_name:
                cycle = get_cycle(entry)
            safety_copy(entry, cycle)
            last_entry = entry_name
        cd('..')

    cd('analysis')

if __name__ == "__main__":
    make_safety_copy()
