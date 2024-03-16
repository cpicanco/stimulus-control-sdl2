from fileutils import cd, list_data_folders, list_files, walk_and_execute_convertion
from converters import convert_data_file, convert_info_file, add_info_to_data_files

def convert():
    print('*****************************   Converting data files...')
    for entry in list_files('.data'):
        convert_data_file(entry)

    print('*****************************   Converting info files...')
    for entry in list_files('.info'):
        convert_info_file(entry)

    print('*****************************   Adding information to data files...')
    for entry in list_files('.data.processed'):
        add_info_to_data_files(entry)

def convert_all():
    cd('..')
    participant_folders = list_data_folders()
    for folder in participant_folders:
        walk_and_execute_convertion(folder, convert)

if __name__ == "__main__":
    convert_all()