from fileutils import cd, list_data_folders, list_files, walk_and_execute
from converters import (
    convert_data_file,
    convert_info_file,
    add_info_to_data_files,
    override_timestamps_file)

def convert(override=True):
    print('*****************************   Converting data files...')
    for entry in list_files('.data'):
        convert_data_file(entry, override=override)

    print('*****************************   Converting info files...')
    for entry in list_files('.info', except_name='.gaze'):
        convert_info_file(entry, override=override)

    print('*****************************   Adding information to data files...')
    for entry in list_files('.data.processed'):
        add_info_to_data_files(entry, override=override)

def convert_all(exclude_list=[], override=False):
    cd('..')

    print('Folders that will be converted:')
    participant_folders = list_data_folders(exclude_list=exclude_list)
    for folder in participant_folders:
        print(folder)
    print('*****************************   Conversion started...')

    for folder in participant_folders:
        walk_and_execute(folder, convert, override)

    cd('analysis')

def convert_one(folder, override=False):
    cd('..')
    print('*****************************   Conversion started...')
    walk_and_execute(folder, convert, override)
    cd('analysis')

def override_timestamps(override=True):
    for entry in list_files('.timestamps'):
        override_timestamps_file(entry, override)

def override_all_timestamps():
    cd('..')

    print('Folders that will be converted:')
    participant_folders = list_data_folders(exclude_list=[])
    for folder in participant_folders:
        print(folder)
    print('*****************************   Overriding all timestamps in safety copies ...')

    for folder in participant_folders:
        walk_and_execute(folder, override_timestamps, True)

    cd('analysis')


if __name__ == "__main__":
    # convert_one('26-MSS', override=True)
    override_all_timestamps()