import pandas as pd

# from converters import convert_data_file
from constants import participants_to_ignore
from study1_constants import foldername as foldername_1
from study2_constants import foldername as foldername_2
from study3_constants import foldername as foldername_3
from study4_constants import foldername as foldername_4

from fileutils import (
    cd,
    load_file,
    list_data_folders,
    change_data_folder_name,
    list_files,
)


from converters import (
    convert_data_file
)

from classes import Information

def fix_session_name_in_info_files(folders=[foldername_1, foldername_2, foldername_3, foldername_4]):
    def fix_all():
        prefixes = {
            'Ciclo1-7': 'Ciclo2-0',
            'Ciclo2-7': 'Ciclo3-0',
            'Ciclo3-7': 'Ciclo4-0',
            'Ciclo4-7': 'Ciclo5-0',
            'Ciclo5-7': 'Ciclo6-0'
        }
        for entry in list_files('.info.processed'):
            info = Information(entry)
            # if session name has the prefix 'Ciclo1-7', replace it with 'Ciclo2-0'
            for prefix in prefixes.keys():
                if prefix in info.session_name:
                    info.session_name = info.session_name.replace(prefix, prefixes[prefix])
                    info.save_to_file()
                    # print(str(info))
                    break

    for foldername in folders:
        change_data_folder_name(foldername)
        for participant_folder in list_data_folders():
            cd(participant_folder, verbose=False)
            cd('analysis', verbose=False)
            for date_folder in list_data_folders():
                cd(date_folder, verbose=False)

                if foldername == foldername_1:
                    fix_all()
                else:
                    for cycle_folder in list_data_folders():
                        cd(cycle_folder, verbose=False)
                        fix_all()
                        cd('..', verbose=False)

                cd('..', verbose=False)
            cd('..', verbose=False)
            cd('..', verbose=False)

# def fix_result_in_info_files():
#     for foldername in [foldername_4]:
#         change_data_folder_name(foldername)
#         for participant_folder in list_data_folders():
#             cd(participant_folder, verbose=False)
#             cd('analysis', verbose=False)
#             for date_folder in list_data_folders():
#                 cd(date_folder, verbose=False)
#                 for entry in list_files('.info.processed'):
#                     info = Information(entry, verbose=False)
#                     # if session name has the prefix 'Ciclo1-7', replace it with 'Ciclo2-0'
#                     result = info.__info_file__.loc[info.__session_result__]
#                     if result.shape[0] > 1:
#                         print(result.iloc[0][1])
#                 cd('..', verbose=False)
#             cd('..', verbose=False)
#             cd('..', verbose=False)




def fix_response_in_data_files(foldername):
    def fix_all():
        for raw_entry in list_files('.data'):
            raw_data = convert_data_file(raw_entry)
            if raw_data is None:
                continue
            processed_entry = raw_entry.replace('.data', '.data.processed')
            df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
            processed_data = load_file(processed_entry)
            processed_data['Response'] = df['Response']
            processed_data.to_csv(processed_entry, sep='\t', index=False)


    change_data_folder_name(foldername)
    for participant_folder in list_data_folders():
        cd(participant_folder)
        cd('analysis')
        for date_folder in list_data_folders():
            cd(date_folder)
            if foldername == foldername_1:
                fix_all()
            else:
                for cycle_folder in list_data_folders():
                    cd(cycle_folder, verbose=False)
                    fix_all()
                    cd('..', verbose=False)
            cd('..', verbose=False)
        cd('..', verbose=False)
        cd('..', verbose=False)

if __name__ == '__main__':
    # fix_session_name_in_info_files([foldername_4])

    # from convertions import (
    #     override_all_timestamps
    # )
    # override_all_timestamps([foldername_4])

    fix_response_in_data_files(foldername_4)