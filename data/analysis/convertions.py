from study1_constants import foldername as foldername_1
from study2_constants import foldername as foldername_2
from study3_constants import foldername as foldername_3
from study4_constants import foldername as foldername_4

from fileutils import (
    cd,
    list_data_folders,
    list_files,
    walk_and_execute,
    walk_and_execute_1,
    change_data_folder_name,
    safety_copy,
    get_cycle)

from converters import (
    convert_data_file,
    convert_info_file,
    add_info_to_data_files,
    override_timestamps_file)

def make_safety_copy(folders=[foldername_1, foldername_2, foldername_3, foldername_4]):
    for foldername in folders:
        change_data_folder_name(foldername)
        data_folder = list_data_folders()

        for data_foldername in data_folder:
            cd(data_foldername)
            last_entry = ''
            for entry in list_files(''):
                entry_name = entry.split('.')[0]
                if last_entry != entry_name:
                    cycle = get_cycle(entry)
                safety_copy(entry, cycle)
                last_entry = entry_name
            cd('..')

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

def convert_all(folders=[foldername_1, foldername_2, foldername_3, foldername_4], exclude_list=[], override=True):
    for foldername in folders:
        change_data_folder_name(foldername)

        print('Folders that will be converted:')
        participant_folders = list_data_folders(exclude_list=exclude_list)
        for folder in participant_folders:
            print(folder)
        print('*****************************   Conversion started...')

        for folder in participant_folders:
            walk_and_execute(folder, convert, override)

def convert_one(folder, override=False):
    cd('..')
    print('*****************************   Conversion started...')
    walk_and_execute(folder, convert, override)
    cd('analysis')

def override_timestamps(override=True):
    for entry in list_files('.timestamps'):
        override_timestamps_file(entry, override)

def override_all_timestamps(folders=[foldername_1, foldername_2, foldername_3, foldername_4]):
    for foldername in folders:
        change_data_folder_name(foldername)

        print('Folders that will be converted:')
        participant_folders = list_data_folders(exclude_list=[])
        for folder in participant_folders:
            print(folder)
        print('*****************************   Overriding all timestamps in safety copies ...')

        for folder in participant_folders:
            if foldername == foldername_1:
                walk_and_execute_1(folder, override_timestamps, True)
            else:
                walk_and_execute(folder, override_timestamps, True)

if __name__ == "__main__":
    # first, make safety copies into {participant-code}/'analysis'
    # make_safety_copy([foldername_4])

    # second, convert data, info files to a regular tagulated format
    # convert_all([foldername_4])

    # optionally, for individual participants
    # convert_one('26-MSS', override=True)

    # third, override all timestamps
    override_all_timestamps([foldername_4])