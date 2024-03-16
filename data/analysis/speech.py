from fileutils import as_data, file_exists
from fileutils import cd, list_files, list_data_folders, load_file
from fileutils import walk_and_execute
from correlation import plot_correlation
from classes import Information

import pandas as pd
import Levenshtein
from unidecode import unidecode_expect_ascii

def data_by_relation(pattern, container):
    for entry in list_files('.info.processed'):
        data_file = as_data(entry, processed=True)
        if not file_exists(data_file):
            print(f'File {data_file} does not exist.')
            continue

        info = Information(entry)
        if info.has_valid_result():
            data = load_file(data_file)
            data = data[data['Relation'].str.match(pattern)]

            if not data.empty:
                print(f'File {entry} has valid result.')
                container.append(data)
            else:
                print(f'File {entry} is empty.')
        else:
            print(f'File {entry} has no valid result.')

def get_probes():
    cd('..')
    container = []
    participant_folders = list_data_folders()
    for folder in participant_folders:
        walk_and_execute(folder, data_by_relation, 'C-D', container)

    return pd.concat(container)

def save_probes():
    data = get_probes()
    data = data[data['HasDifferentialReinforcement'] == False]
    column = data['Name']
    data.drop(columns='Name', inplace=True)
    data['Name'] = column

    cd('analysis')
    cd('output')
    data.to_csv('probes_CD.data', sep='\t' , index=False)

def validated_speech(word):
    if not isinstance(word, str):
        word = ''
    return unidecode_expect_ascii(word.strip().lower().replace(' ', ''))

def similarity(row):
    return Levenshtein.ratio(row['Name'], validated_speech(row['Speech-2']))

def result(row):
    if row['Name'] == validated_speech(row['Speech-2']):
        return 'Hit'
    return 'Miss'

def calculate_similarity():
    cd('..')
    cd('analysis')
    cd('output')
    data = pd.read_csv('probes_CD.transcripted.data', sep='\t')
    data['Levenshtein'] = data.apply(lambda row: similarity(row), axis=1)
    data['Result'] = data.apply(lambda row: result(row), axis=1)
    data['Latency'] = data.apply(lambda row: row['Latency'].replace(',', '.'), axis=1)
    data['Participant'] = data.apply(lambda row: row['Participant'].replace('\\', ''), axis=1)
    data.to_csv('probes_CD.transcripted.data.processed', sep='\t' , index=False)

def correlate_latency_levenshtein(do_global_analysis=False):
    cd('..')
    participants = list_data_folders()
    print(participants)
    cd('analysis')
    cd('output')
    all_data = pd.read_csv('probes_CD.transcripted.data.processed', sep='\t')

    if do_global_analysis:
        # filter data by word name
        data = all_data[all_data['Name'].str.match(r'(bena|falo)')]
        data = data.sort_values(by=['Cycle.ID', 'Time'])
        plot_correlation(data['Levenshtein'], data['Latency'], 'Levenshtein', 'Latency', 'Bena e Falo')
        plot_correlation(data.index, data['Levenshtein'], 'Trial', 'Levenshtein', 'Bena e Falo')
        plot_correlation(data.index, data['Latency'], 'Trial', 'Latency', 'Bena e Falo')
    else:
        for participant in participants:
            # filter data by participant
            data = all_data[all_data['Participant'] == participant]
            # sort data by Cycle and Time
            data = data.sort_values(by=['Cycle.ID', 'Time'])
            data = data[data['Name'].str.match(r'(bena|falo)')]
            # filter by word
            plot_correlation(data['Levenshtein'], data['Latency'], 'Levenshtein', 'Latency', participant+'- Bena e Falo')
            plot_correlation(data.index, data['Levenshtein'], 'Index', 'Levenshtein', participant+'- Bena e Falo')
            plot_correlation(data.index, data['Latency'], 'Index', 'Latency', participant+' - Bena e Falo')

def override_CD_probes_in_data_file(must_not_override=True):
    cd('output')
    data = pd.read_csv('probes_CD.transcripted.data.processed', sep='\t')
    print('----------------------------- override data files and creating probes files')
    cd('..')
    cd('..')
    participant_folders = list_data_folders()
    for participant_folder in participant_folders:
        cd(participant_folder)
        cd('analysis')
        safety_copy_data_folders = list_data_folders()
        for data_folder_by_day in safety_copy_data_folders:
            cd(data_folder_by_day)

            filtered_data = None
            for entry in list_files('.info.processed'):
                data_file = as_data(entry, processed=True)
                if not file_exists(data_file):
                    continue

                info = Information(entry)
                if info.has_valid_result():
                    if not '-CD-' in info.session_name:
                        continue

                    if file_exists(data_file.replace('.data', '.probes')):
                        if must_not_override:
                            continue

                    data_to_override = load_file(data_file)

                    filtered_data = data[data['File'] == data_file]

                    participant = filtered_data['Participant'].str.replace(r'\\', '')
                    filtered_data = filtered_data[participant == participant_folder]

                    # check if count is the same
                    if filtered_data.shape[0] != data_to_override.shape[0]:
                        print(f'Count is different for {data_file}')
                        raise Exception('Count is different')
                    else:
                        data_to_override['Result'] = filtered_data['Result'].values
                        data_to_override['Response'] = filtered_data['Speech-2'].values
                        data_to_override.to_csv(data_file, sep='\t', index=False)
                        filtered_data['Levenshtein'].to_csv(data_file.replace('.data', '.probes'), sep='\t', index=False)
            cd('..')
        cd('..')
        cd('..')

if __name__ == "__main__":
    # save_probes()
    # calculate_similarity()
    # correlate_latency_levenshtein()
    override_CD_probes_in_data_file()