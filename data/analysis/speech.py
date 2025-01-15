from fileutils import as_data, file_exists
from fileutils import cd, list_files, list_data_folders, load_file
from fileutils import walk_and_execute, get_data_folders
from correlation import plot_correlation, plot_correlation_2
from classes import Information

import pandas as pd
import Levenshtein
from unidecode import unidecode_expect_ascii
from anonimizator import anonimize

target_speech = 'Speech-3'

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

def load_probes_file(filename):
    dtype_dict = {
        'Report.Timestamp': str,
        'Session.Trial.UID': int,
        'Session.Block.UID': int,
        'Session.Block.Trial.UID': int,
        'Session.Block.ID': int,
        'Session.Block.Trial.ID': int,
        'Session.Block.Name': str,
        'Trial.ID': int,
        'Cycle.ID': int,
        'Relation': str,
        'Comparisons': int,
        'Result': str,
        'CounterHit': int,
        'CounterHit.MaxConsecutives': int,
        'CounterMiss': int,
        'CounterMiss.MaxConsecutives': int,
        'CounterNone': int,
        'CounterNone.MaxConsecutives': int,
        'Sample-Position.1': str,
        'Comparison-Position.1': str,
        'Comparison-Position.2': str,
        'Comparison-Position.3': str,
        'Response': str,
        'HasDifferentialReinforcement': bool,
        'Latency': str,
        'Participant': str,
        'Condition': int,
        'Date': str,
        'Time': str,
        'File': str,
        'Name': str,
        'ID': int,
        'Speech': str,
        'Speech-2': str,
        'Speech-3': str
    }
    return pd.read_csv(filename, sep='\t', header=0, engine='python', dtype=dtype_dict)

def save_probes_file(filename, data):
    data.to_csv(filename, sep='\t', index=False)

def get_probes():
    participant_folders = get_data_folders()
    container = []
    for folder in participant_folders:
        walk_and_execute(folder, data_by_relation, 'C-D', container)
    return pd.concat(container)

def save_probes_by_participant(should_fix_cycles):
    data = get_probes()
    # sort first by Date/Time, from oldest to newest
    # but convert to DateTime first
    # data['DateTime'] = pd.to_datetime(f"{data['Date']} {data['Time']}", format='%d/%m/%Y %H:%M:%S')
    # data = data.sort_values(by=['DateTime', ''])
    data = data[data['HasDifferentialReinforcement'] == False]
    column = data['Name']
    data.drop(columns='Name', inplace=True)
    data['Name'] = column

    cd('analysis')
    cd('output')

    # get unique names from  participant column
    participants = data['Participant'].unique()
    # remove nan from participants
    participants = [x for x in participants if str(x) != 'nan']
    for participant in participants:
        # filter data per participant
        filtered_data = data[data['Participant'] == participant]
        # add a new column with incremental ID
        filtered_data = filtered_data.copy()
        filtered_data['ID'] = range(1, len(filtered_data) + 1)

        # save data to participant file
        participant = participant.replace('\\', '').replace('-', '_')
        # check if participant is float
        filename = f'probes_CD_{participant}.data'
        # check if file exists
        if file_exists(filename):
            # load file
            existing_data = load_probes_file(filename)
            # if same lenght, do nothing, there is no data to append
            if existing_data.shape[0] == filtered_data.shape[0]:
                print(f'File {filename} already exists and has the same length. Skipping...')
                if should_fix_cycles:
                    print(f'Fixing cycles for {filename}...')
                    existing_data['Cycle.ID'] =  filtered_data['Cycle.ID'].reset_index(drop=True)
                    existing_data.to_csv(filename, sep='\t', index=False)
                continue

            print(f'File {filename} already exists. Appending...')
            # concatenate data and keep only unique rows
            filtered_data = pd.concat([existing_data, filtered_data])
            # drop duplicates in-place
            filtered_data.drop_duplicates(subset=['ID'], keep='first', inplace=True)
        else:
            print(f'File {filename} does not exist. Creating...')
        # add 'Speech', target_speech if not exists
        if 'Speech' not in filtered_data.columns:
            filtered_data['Speech'] = ''
        if 'Speech-2' not in filtered_data.columns:
            filtered_data['Speech-2'] = ''
        if 'Speech-3' not in filtered_data.columns:
            filtered_data['Speech-3'] = ''
        filtered_data.to_csv(filename, sep='\t' , index=False)

    cd('..')

def load_all_probes():
    participant_folders = get_data_folders(anonimized=True)
    cd('analysis')
    cd('output')
    container = []
    for participant in participant_folders:
        participant = participant.replace('\\', '').replace('-', '_')
        try:
            filename = f'probes_CD_{participant}.data'
            data = load_probes_file(filename)
            print(f'File {filename} loaded.')
        except FileNotFoundError:
            print(f'File {filename} not found.')
            continue
        container.append(data)
    return pd.concat(container)

def concatenate_probes(filename='probes_CD.data'):
    data = load_all_probes()
    save_probes_file(filename, data)
    print(f'All probes concatenated to file: {filename}')

def validated_speech(word):
    if not isinstance(word, str):
        word = ''
    return unidecode_expect_ascii(word.strip().lower().replace(' ', ''))

def similarity(row):
    return Levenshtein.ratio(row['Name'], validated_speech(row[target_speech]))

def result(row):
    if row['Name'] == validated_speech(row[target_speech]):
        return 'Hit'
    return 'Miss'

def calculate_similarity():
    cd('..')
    cd('analysis')
    cd('output')
    filename = 'probes_CD.data'
    data = load_probes_file(filename)
    data['Levenshtein'] = data.apply(lambda row: similarity(row), axis=1)
    data['Result'] = data.apply(lambda row: result(row), axis=1)
    data['Latency'] = data.apply(lambda row: row['Latency'].replace(',', '.'), axis=1)
    data['Participant'] = data.apply(lambda row: row['Participant'].replace('\\', ''), axis=1)
    data.to_csv('probes_CD.data.processed', sep='\t' , index=False)
    # print a message indicating that the process has finnished
    print('Levenshtein similarity calculated.')

def correlate_latency_levenshtein(do_global_analysis=False):
    cd('..')
    participants = list_data_folders(exclude_list=['22-GLB'])
    print(participants)
    cd('analysis')
    cd('output')
    filename = 'probes_CD.data.processed'
    all_data = load_probes_file(filename)

    if do_global_analysis:
        # filter data by word name
        data = all_data[all_data['Name'].str.match(r'(bena|falo)')]
        # data = data.sort_values(by=['Date', 'Time'])
        # plot_correlation(data['Levenshtein'], data['Latency'], 'Levenshtein', 'Latency', 'Bena e Falo')
        # plot_correlation(data.index, data['Levenshtein'], 'Trial', 'Levenshtein', 'Bena e Falo')
        plot_correlation(data.index, data['Latency'], 'Trial', 'Latency', 'Bena e Falo')
    else:
        for participant in participants:
            # print participant name and message
            print(f'Analysing participant {participant}.....')
            name = anonimize(participant, as_path=False)
            # filter data by participant
            data = all_data[all_data['Participant'] == name]

            # sort data by Cycle and Time
            # data = data.sort_values(by=['Date'])

            # 4 per cycle
            regular_expression1 = r'(bena|falo)'
            words1 = 'Bena/Falo'

            # 4 per cycle
            regular_expression2 = r'(nibe|lofi|bofi|nale|leba|nofa|bona|lefi|fabe|nilo|febi|lano)'
            words2 = 'Nibe/Lofi (etc)'

            data = data[data['Condition'] == 5]
            data['Latency'] = data['Latency'].map(lambda x: x.replace(',', '.'))
            data['Latency'] = data['Latency'].astype(float)

            data1 = data[data['Name'].str.match(regular_expression1)]
            data1.reset_index(inplace=True)
            data1.index = data1.index + 1

            data2 = data[data['Name'].str.match(regular_expression2)]
            data2.reset_index(inplace=True)
            data2.index = data2.index + 1
            # print(data['Latency'])
            # plot_correlation(data.index, data['Levenshtein'], 'Trials', 'Levenshtein index', name+'- Bena e Falo')
            # plot_correlation(data.index, data['Latency'], 'Trials', 'Latency', name+' - '+ words, save=True)
            plot_correlation_2(data1.index, data1['Latency'], data2.index, data2['Latency'],
                               'Trials', 'Latency',
                               name=name, title1=words1, title2=words2, save=True)

def override_CD_probes_in_data_file(must_not_override=True):
    cd('output')
    filename = 'probes_CD.data.processed'
    data = load_probes_file(filename)
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

            for cycle in list_data_folders():
                cd(cycle)
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
                        # print a message idenifying participant, folder, and file
                        print(f'Participant: {participant_folder}, Folder: {data_folder_by_day}, File: {data_file}')
                        data_to_override = load_file(data_file)

                        filtered_data = data[data['File'] == data_file]

                        participant = filtered_data['Participant'].str.replace(r'\\', '')
                        filtered_data = filtered_data[participant == anonimize(participant_folder, as_path=False)]

                        # check if count is the same
                        if filtered_data.shape[0] != data_to_override.shape[0]:
                            print(f'Count is different for {data_file}')
                            raise Exception('Count is different')
                        else:
                            data_to_override['Result'] = filtered_data['Result'].values
                            data_to_override['Response'] = filtered_data['Speech-3'].values
                            data_to_override.to_csv(data_file, sep='\t', index=False)
                            filtered_data['Levenshtein'].to_csv(data_file.replace('.data', '.probes'), sep='\t', index=False)
                cd('..')
            cd('..')
        cd('..')
        cd('..')


if __name__ == "__main__":
    override_CD_probes_in_data_file(must_not_override=False)