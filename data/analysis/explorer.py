import os
import pickle
from collections import Counter

import pandas as pd
import numpy as np

# from converters import convert_data_file
from constants import participants_to_ignore
from study1_constants import foldername as foldername_1
from study2_constants import foldername as foldername_2
from study3_constants import foldername as foldername_3
from player_synchronizer import Synchronizer
from metadata import Metadata
from words import words_per_file
from fileutils import (
    cd,
    # load_file,
    list_data_folders,
    change_data_folder_name,
    list_files,
    cache_folder
)

from classes import Information

# def fix_session_name_in_info_files():
#     prefixes = {
#         'Ciclo1-7': 'Ciclo2-0',
#         'Ciclo2-7': 'Ciclo3-0',
#         'Ciclo3-7': 'Ciclo4-0',
#         'Ciclo4-7': 'Ciclo5-0',
#         'Ciclo5-7': 'Ciclo6-0'
#     }

#     for foldername in [foldername_1]:
#         change_data_folder_name(foldername)
#         for participant_folder in list_data_folders():
#             cd(participant_folder, verbose=False)
#             cd('analysis', verbose=False)
#             for date_folder in list_data_folders():
#                 cd(date_folder, verbose=False)
#                 for entry in list_files('.info.processed'):
#                     info = Information(entry)
#                     # if session name has the prefix 'Ciclo1-7', replace it with 'Ciclo2-0'
#                     for prefix in prefixes.keys():
#                         if prefix in info.session_name:
#                             info.session_name = info.session_name.replace(prefix, prefixes[prefix])
#                             info.save_to_file()
#                             # print(str(info))
#                             break
#                 cd('..', verbose=False)
#             cd('..', verbose=False)
#             cd('..', verbose=False)

# def fix_result_in_info_files():
#     for foldername in [foldername_1]:
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



#
# from convertions import override_all_timestamps()
#

# def fix_response_in_data_files(foldername):
#     def fix_all():
#         for raw_entry in list_files('.data'):
#             raw_data = convert_data_file(raw_entry)
#             if raw_data is None:
#                 continue
#             processed_entry = raw_entry.replace('.data', '.data.processed')
#             df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
#             processed_data = load_file(processed_entry)
#             processed_data['Response'] = df['Response']
#             processed_data.to_csv(processed_entry, sep='\t', index=False)


#     change_data_folder_name(foldername)
#     for participant_folder in list_data_folders():
#         cd(participant_folder)
#         cd('analysis')
#         for date_folder in list_data_folders():
#             cd(date_folder)
#             if foldername == foldername_1:
#                 fix_all()
#             else:
#                 for cycle_folder in list_data_folders():
#                     cd(cycle_folder, verbose=False)
#                     fix_all()
#                     cd('..', verbose=False)
#             cd('..', verbose=False)
#         cd('..', verbose=False)
#         cd('..', verbose=False)


def append_session_name(names):
    for entry in list_files('.info.processed'):
        info = Information(entry, verbose=False)
        if info.has_valid_result():
            names.append(info.session_name)

def unique_session_names(foldername):
    change_data_folder_name(foldername)
    for participant_folder in list_data_folders():
        unique_names = []

        cd(participant_folder, verbose=False)
        cd('analysis', verbose=False)
        for date_folder in list_data_folders():
            cd(date_folder, verbose=False)
            if foldername == foldername_1:
                append_session_name(unique_names)
            else:
                for cycle_folder in list_data_folders():
                    cd(cycle_folder, verbose=False)
                    append_session_name(unique_names)
                    cd('..', verbose=False)
            cd('..', verbose=False)
        cd('..', verbose=False)
        cd('..', verbose=False)

    unique_names = list(set(unique_names))
    unique_names.sort()
    return unique_names

date_format = '%Y-%m-%d'

def find_duplicates(sessions):
    return [{item : count} for item, count in Counter(sessions).items() if count > 1]

def load_from_cache(foldername, participant, entry):
    code = os.path.splitext(os.path.splitext(entry)[0])[0]
    filename = '_'.join([
        foldername,
        participant.replace('-', '_'), code])
    return load_pickle(filename)

def append_session(sessions, use_cache=False, foldername=''):
    for entry in list_files('.info.processed'):
        info = Information(entry, verbose=False)
        if info.has_valid_result():
            participant_name = info.participant_name.replace('\\', '')
            if participant_name == '3-PVV':
                if entry == '034.info.processed':
                    continue
            if use_cache:
                if foldername == '':
                    raise Exception('Foldername is empty.')
                if 'Pre-treino' in info.session_name:
                    session = []
                else:
                    session = load_from_cache(foldername, participant_name, entry)
            else:
                session = info
            sessions.append({
                'session' : session,
                'participant_name': participant_name,
                'session_name': info.session_name,
                'session_date_time': \
                    info.start_date.to_string(date_format) + \
                    '\t' + \
                    str(info.start_time),
                'session_duration': info.duration
            })

def inspect_session_names_ordered_by_date(foldername):
    participants = []
    change_data_folder_name(foldername)
    for participant_folder in list_data_folders():
        if participant_folder in participants_to_ignore:
            continue
        cd(participant_folder, verbose=False)
        cd('analysis', verbose=False)
        sessions = []
        for date_folder in list_data_folders():
            cd(date_folder, verbose=False)
            if foldername == foldername_1:
                append_session(sessions, True, foldername)
            else:
                for cycle_folder in list_data_folders():
                    cd(cycle_folder, verbose=False)
                    append_session(sessions, True, foldername)
                    cd('..', verbose=False)
            cd('..', verbose=False)
        sessions.sort(key=lambda x: x['session_date_time'])
        participants.append(sessions)
        cd('..', verbose=False)
        cd('..', verbose=False)

    return participants

def inspect_total_sessions_by_participant(foldername):
    unique_sessions = unique_session_names(foldername)
    participants = []
    change_data_folder_name(foldername)
    for participant_folder in list_data_folders():
        sessions = []
        cd(participant_folder, verbose=False)
        cd('analysis', verbose=False)
        for date_folder in list_data_folders():
            cd(date_folder, verbose=False)
            if foldername == foldername_1:
                append_session(sessions)
            else:
                for cycle_folder in list_data_folders():
                    cd(cycle_folder, verbose=False)
                    append_session(sessions)
                    cd('..', verbose=False)
            cd('..', verbose=False)
        participants.append({
            'participant_name': participant_folder,
            'sessions': len(sessions),
            'missing_sessions': [f for f in unique_sessions if f not in [s['session_name'] for s in sessions]],
            'duplicate_sessions': find_duplicates([s['session_name'] for s in sessions]),
            'total_duration': sum([s['session_duration'] for s in sessions])
        })
        cd('..', verbose=False)
        cd('..', verbose=False)

    return participants

def process_sources(sources, save_trials):
    calculate_fixation_mapping = {
        foldername_1: False,
        foldername_2: True,
        foldername_3: False
    }
    for entry in list_files('.info.processed'):
        info = Information(entry)
        participant = info.participant_name.replace('\\', '')
        foldername = sources[participant]['metadata']['study']
        if info.has_valid_result():
            design_file = info.session_name
            cycle = design_file.split('-')[0].replace('Ciclo', '')
            date = info.start_date.to_string(date_format)

            if foldername == foldername_1:
                path = os.path.join(participant, 'analysis', date)
            else:
                path = os.path.join(participant, 'analysis', date, cycle)

            if participant == '3-PVV' \
            and cycle == '5' \
            and date == '2024-07-25' \
            and design_file == 'Ciclo5-0-Sondas-CD-Palavras-12-ensino-8-generalizacao' \
            and entry == '034.info.processed':
                print('This session has no eye tracking data. Skipping...')
                continue

            code = os.path.splitext(os.path.splitext(entry)[0])[0]
            sources[participant][design_file].append({
                'info': info,
                'words': words_per_file[design_file],
                'name': design_file,
                'participant': participant,
                'date': date,
                'cycle': cycle,
                'path': path,
                'code': code,
            })

            if save_trials:
                filename = '_'.join([
                            foldername,
                            participant.replace('-', '_'), code])
                session = Synchronizer(code)
                trials = session.trial_filter()

                save_pickle(filename+'_trials', trials.DataFrame)

                # if calculate_fixation_mapping[foldername]:
                #     save_pickle(filename+'_processed_fixations', trials.processed_fixations())
                #     save_pickle(filename+'_raw_fixations', trials.raw_fixations())

                del session
                del trials

def save_participant_sources(foldername, save_trials=False):
    change_data_folder_name(foldername)

    filename = foldername +'_design_files'
    cache_foldername = cache_folder()
    cache_filename = os.path.join(cache_foldername, filename)
    if os.path.exists(cache_filename+'.pkl'):
        design_files = load_pickle(filename)
    else:
        design_files = unique_session_names(foldername)
        save_pickle(filename, design_files)

    sources = {}
    for participant in list_data_folders():

        if participant == '2-FEB' \
        or participant == '6-RIC' \
        or participant == '11-VSC' \
        or participant == '18-JEG' \
        or participant == '21-NLC' \
        or participant == '24-VCC':
            print(f'Participant {participant} has incomplete data. Skipping...')
            continue

        sources[participant] = {}
        for design_file in design_files:
            # if key does not exist, create it
            if sources[participant].get(design_file) is None:
                sources[participant][design_file] = []

        cd(os.path.join(participant, 'analysis'))

        # load metadata
        sources[participant]['metadata'] = Metadata().items
        sources[participant]['metadata']['study'] = foldername

        for date_folder in list_data_folders():
            cd(date_folder)
            if foldername == foldername_1:
                process_sources(sources, save_trials)
            else:
                for cycle_folder in list_data_folders():
                    cd(cycle_folder)
                    process_sources(sources, save_trials)
                    cd('..')
            cd('..')
        cd(os.path.join('..','..'))


        # count how many design files are for this participant, we need 50
        if len(sources[participant]) < 50:
            print(f'Warning: {participant} has only {len(sources[participant])} design files')

        for design_file in sources[participant]:
            if design_file == 'metadata':
                continue

            number = len(sources[participant][design_file])
            if number == 0:
                print(f'Warning: {participant} has no data for {design_file}')
            elif number > 1:
                print(f'Warning: {participant} has multiple data (n={number}) for {design_file}')
                for data in sources[participant][design_file]:
                    print(f'  - {data["date"]}, {data["cycle"]}, {data["code"]}')

    filename = foldername
    cache_foldername = cache_folder()
    cache_filename = os.path.join(cache_foldername, filename)
    design_files = unique_session_names(foldername)
    save_pickle(filename, sources)

def load_pickle(filename):
    output_path = os.path.join(cache_folder(), filename +'.pkl')
    with open(output_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(filename, data):
    folder = cache_folder()
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = os.path.join(folder, filename +'.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_participant_sources(foldername) -> dict:
    # load sources
    output_path = os.path.join(cache_folder(), foldername+'.pkl')
    if not os.path.exists(output_path):
        print(f"Cache file '{output_path}' not found.")
        return {}

    with open(output_path, 'rb') as f:
        return pickle.load(f)

def calculate_fixations_duration(df: pd.DataFrame):
    duration = 0
    for _, fixation_data in df.groupby('FPOGID'):
        duration =+ fixation_data['FPOGD'].max()

    if duration > 1:
        pass
    return duration

def export(data : dict, filename='data'):
    previous_length = None
    for column, row in data.items():
        new_length = len(row)
        if previous_length is not None and previous_length != new_length:
            print(f'Warning: {column} has different length')
        print(column, new_length)
        previous_length = new_length

    df = pd.DataFrame(data)
    folder = cache_folder()
    df.to_csv(os.path.join(folder, filename+'.csv'), index=False)


prefixes = ['sample', 'comparisons']

def new_dict(eye_measurements=False):
    d = {
        'uid':[],
        'study': [],
        'duration':[],
        'uid_in_session':[],
        'trial_id':[],
        'cycle':[],
        'relation':[],
        'condition':[],
        'result':[],
        'response' : [],
        'file':[],
        'file_date':[],
        'file_start_time':[],
        'has_differential_reinforcement':[],
        'participant':[],
        'latency':[],
        'reference_word':[]
    }
    for prefix in prefixes:
        d[f'{prefix}_duration'] = []
        if prefix == 'sample':
            maximum_range = 1
        elif prefix == 'comparisons':
            maximum_range = 4

        for i in range(0, maximum_range):
            n = i + 1
            d[f'{prefix}{n}'] = []


    if eye_measurements:
        d['fixations_count'] = []
        d['fixations_duration'] = []
        for prefix in prefixes:
            d[f'{prefix}_count'] = []
            d[f'{prefix}_switchings'] = []
            if prefix == 'sample':
                maximum_range = 1
            elif prefix == 'comparisons':
                maximum_range = 4

            for i in range(0, maximum_range):
                n = i + 1
                d[f'{prefix}{n}_fixations_count'] = []
                d[f'{prefix}{n}_fixations_duration'] = []
                d[f'{prefix}{n}_letters_switchings'] = []

                for j in range(0, 4):
                    k = j + 1
                    d[f'{prefix}{n}_letter{k}_fixations_count'] = []
                    d[f'{prefix}{n}_letter{k}_fixations_duration'] = []

    return d


def export_to_csv_2():
    dataframe_dict = new_dict(True)

    sources = load_participant_sources(foldername_2)
    for participant in sources:
        uid = 1
        participant_metadata = sources[participant].pop('metadata')
        # del sources[participant]['metadata']
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    filename = '_'.join([
                        participant_metadata['study'],
                        participant.replace('-', '_'), source['code']])
                    print(filename)

                    trials = load_pickle(filename)

                    for trial in trials:
                        relation = trial['relation']
                        dataframe_dict['uid'].append(uid)
                        dataframe_dict['study'].append(participant_metadata['study'])
                        dataframe_dict['duration'].append(trial['duration'])
                        dataframe_dict['uid_in_session'].append(trial['trial_uid_in_session'])
                        dataframe_dict['trial_id'].append(trial['trial_id'])
                        dataframe_dict['cycle'].append(trial['cycle'])
                        dataframe_dict['relation'].append(relation)
                        dataframe_dict['condition'].append(trial['condition'])
                        dataframe_dict['result'].append(trial['result'])
                        dataframe_dict['response'].append(trial['response'])
                        dataframe_dict['file'].append(trial['file'])
                        dataframe_dict['file_date'].append(trial['file_date'])
                        dataframe_dict['file_start_time'].append(trial['file_start_time'])
                        dataframe_dict['has_differential_reinforcement'].append(trial['has_differential_reinforcement'])
                        dataframe_dict['participant'].append(trial['participant'])
                        dataframe_dict['latency'].append(trial['latency'])
                        dataframe_dict['reference_word'].append(trial['reference_word'])

                        fixations, timestamps = trial['fixations']
                        fixations_count = len(fixations)
                        fixations_duration = calculate_fixations_duration(fixations)
                        dataframe_dict['fixations_count'].append(fixations_count)
                        dataframe_dict['fixations_duration'].append(fixations_duration)

                        for prefix in prefixes:
                            # comparisons
                            if prefix == 'sample':
                                maximum_range = 1
                            elif prefix == 'comparisons':
                                maximum_range = 4

                            # print(relation, prefix, trial['trial_uid_in_session'])
                            if trial[f'{prefix}_duration'] is None:
                                duration = np.nan
                            else:
                                duration = trial[f'{prefix}_duration']

                            if trial[f'{prefix}_fixations'] is None:
                                fixations_by_word = []
                                word_measures = {'switchings': np.nan}
                            else:
                                fixations_by_word, word_measures = trial[f'{prefix}_fixations']

                            count = len(fixations_by_word)
                            if count == 0:
                                switchings = 0
                            else:
                                switchings = word_measures['switchings']

                            dataframe_dict[f'{prefix}_count'].append(count)
                            dataframe_dict[f'{prefix}_duration'].append(duration)
                            dataframe_dict[f'{prefix}_switchings'].append(switchings)

                            fixations_by_word.extend([np.nan] * (maximum_range - len(fixations_by_word)))
                            for i, word in enumerate(fixations_by_word):
                                n = i + 1
                                if word is np.nan:
                                    dataframe_dict[f'{prefix}{n}'].append(np.nan)
                                    dataframe_dict[f'{prefix}{n}_fixations_count'].append(np.nan)
                                    dataframe_dict[f'{prefix}{n}_fixations_duration'].append(np.nan)
                                    dataframe_dict[f'{prefix}{n}_letters_switchings'].append(np.nan)

                                    for j, nan in enumerate([np.nan] * 4):
                                        k = j + 1
                                        dataframe_dict[f'{prefix}{n}_letter{k}_fixations_count'].append(nan)
                                        dataframe_dict[f'{prefix}{n}_letter{k}_fixations_duration'].append(nan)
                                else:
                                    if 'letter_fixations' not in word:
                                        fixations_by_letter = [np.nan] * 4
                                        letter_measures = {'switchings': np.nan}
                                    else:
                                        fixations_by_letter, letter_measures = word['letter_fixations']

                                    dataframe_dict[f'{prefix}{n}'].append(word['text'])
                                    dataframe_dict[f'{prefix}{n}_fixations_count'].append(word['fixations_count'])
                                    dataframe_dict[f'{prefix}{n}_fixations_duration'].append(calculate_fixations_duration(word['contained_fixations']))
                                    dataframe_dict[f'{prefix}{n}_letters_switchings'].append(letter_measures['switchings'])

                                    for j, letter in enumerate(fixations_by_letter):
                                        k = j + 1
                                        if letter is np.nan:
                                            dataframe_dict[f'{prefix}{n}_letter{k}_fixations_count'].append(np.nan)
                                            dataframe_dict[f'{prefix}{n}_letter{k}_fixations_duration'].append(np.nan)
                                        else:
                                            dataframe_dict[f'{prefix}{n}_letter{k}_fixations_count'].append(letter['fixations_count'])
                                            dataframe_dict[f'{prefix}{n}_letter{k}_fixations_duration'].append(calculate_fixations_duration(letter['contained_fixations']))
                        uid += 1

    export(dataframe_dict, foldername_2)


def export_to_csv():
    study_uid_map = {
        foldername_1: 1,
        foldername_2: 2,
        foldername_3: 3,
    }
    foldernames = [foldername_1, foldername_2, foldername_3]
    exporting1 = []
    exporting2 = []
    for foldername in foldernames:
        sources = load_participant_sources(foldername)
        for participant in sources:
            participant_metadata = sources[participant].pop('metadata')
            for design_file in sources[participant]:
                for source in sources[participant][design_file]:
                    if 'Pre-treino' not in source['name']:
                        study = participant_metadata['study']
                        filename = '_'.join([
                            study,
                            participant.replace('-', '_'), source['code']]) + '_trials'
                        print(filename)
                        trials = load_pickle(filename)
                        trials['Study'] = study_uid_map[study]
                        exporting1.append(trials)
            exporting2.append(participant_metadata)

    del sources
    del participant
    del participant_metadata
    del filename
    del foldername

    df = pd.concat(exporting1, ignore_index=True)

    df['Sample-Position.1'] = df['Sample-Position.1'].str.split('-').str[0]
    df['Sample_Position'] = df['Sample-Position.1'].str.split('-').str[1]

    for i in range(1, 4):
        column1 = f'Comparison_{i}_Position'
        column2 = f'Comparison-Position.{i}'

        df[column2] = df[column2].str.split('-').str[0]
        df[column1] = df[column2].str.split('-').str[1]

    df['Sample-Position.1'] = df['Sample.Figure']

    df['Session_Date_Time'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)

    df.drop(columns=[
        'Date',
        'Time',
        'Session.Block.UID',
        'Session.Block.Trial.UID',
        'Session.Block.ID',
        'Session.Block.Trial.ID',
        'Session.Block.Name',
        'CounterHit',
        'CounterHit.MaxConsecutives',
        'CounterMiss',
        'CounterMiss.MaxConsecutives',
        'CounterNone',
        'CounterNone.MaxConsecutives',
        'Sample.Figure'],
        axis=1, inplace=True)


    df['Participant_UID'] = 0

    column_rename_map = {
        # study level
        'Study': 'Study_UID',
        'Trial.ID': 'Trial_UID_In_Study',
        'Participant': 'Participant_UID_In_Study',

        # Participant-level
        'Participant_UID': 'Participant_UID',
        'Cycle.ID': 'Cycle_UID_In_Participant',
        'Condition': 'Condition_UID_In_Participant',
        'File': 'Session_UID_In_Participant',

        # Session-level
        'Session_Date_Time': 'Session_Date_Time',
        'Session.Trial.UID': 'Trial_UID_In_Session',
        'Report.Timestamp': 'Trial_Timestamp_In_Session',

        # trial level
        'Name': 'Trial_Word',
        'Relation': 'Trial_Relation',
        'Trial.Duration': 'Trial_Duration',
        'Comparisons': 'Trial_Comparison_Count',
        'Result': 'Trial_Participant_Response_Outcome',
        'Response': 'Trial_Participant_Response',
        'Latency': 'Trial_Participant_Response_Latency',
        'HasDifferentialReinforcement': 'Trial_Differential_Reinforcement_Flag',
        'Sample-Position.1': 'Trial_Sample_Stimulus',
        'Sample_Position': 'Trial_Sample_Stimulus_Position',
        'Comparison-Position.1': 'Trial_Comparison_Stimulus_1',
        'Comparison_1_Position': 'Trial_Comparison_Stimulus_1_Position',
        'Comparison-Position.2': 'Trial_Comparison_Stimulus_2',
        'Comparison_2_Position': 'Trial_Comparison_Stimulus_2_Position',
        'Comparison-Position.3': 'Trial_Comparison_Stimulus_3',
        'Comparison_3_Position': 'Trial_Comparison_Stimulus_3_Position',
        'Sample.Duration': 'Trial_Sample_Stimulus_Duration',
        'Comparisons.Duration': 'Trial_Comparison_Stimuli_Duration',
    }
    df.rename(columns=column_rename_map, inplace=True)

    df['Trial_Duration'] = df['Trial_Duration'].round(4)
    df['Trial_Timestamp_In_Session'] = df['Trial_Timestamp_In_Session'].round(4)
    df['Trial_Participant_Response_Latency'] = df['Trial_Participant_Response_Latency'].round(4)
    df['Trial_Sample_Stimulus_Duration'] = df['Trial_Sample_Stimulus_Duration'].round(4)
    df['Trial_Comparison_Stimuli_Duration'] = df['Trial_Comparison_Stimuli_Duration'].round(4)

    df.loc[:, 'p_num'] = df['Participant_UID_In_Study'].str.extract(r'(\d+)').astype(int)
    df['Participant_UID_In_Study'] = 'P'+ df['p_num'].astype(str)
    df.drop(columns=['p_num'], inplace=True)

    # joine Date and Time in a single column
    df['Session_Date_Time'] = pd.to_datetime(df['Session_Date_Time'], format='%d/%m/%Y %H:%M:%S')
    # sort by date, time, participant, session, trial
    df = df.sort_values(by=['Session_Date_Time'])

    df['p_num'] = df['Study_UID'].astype(str) + df['Participant_UID_In_Study'].astype(str)
    df['Participant_UID'] = (pd.factorize(df['p_num'])[0] + 1).astype(int)
    df.drop(columns=['p_num'], inplace=True)


    df.loc[:, 'file_num'] = df['Session_UID_In_Participant'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
    df.loc[:, 'Session_UID_In_Participant'] = (df.groupby('Participant_UID')['file_num']
                    .rank(method='dense')
                    .astype(int))
    df.drop(columns=['file_num'], inplace=True)
    max_sessions = df.groupby('Participant_UID')['Session_UID_In_Participant'].transform('max')
    df.loc[df['Session_UID_In_Participant'] == max_sessions, 'Cycle_UID_In_Participant'] = 7

    # make sure that Trial_UID_In_Session is int
    df['Trial_UID_In_Session'] = df['Trial_UID_In_Session'].astype(int)
    # sort by date, time, participant, session, trial
    df = df.sort_values(by=['Study_UID', 'Participant_UID', 'Session_UID_In_Participant', 'Trial_UID_In_Session'])
    # assign 7 to empty CONDITION_UID_In_Participant column in study 3
    df.loc[df['Condition_UID_In_Participant'] == '', 'Condition_UID_In_Participant'] = 7
    # reorder columns to match column_rename_map order
    df = df[column_rename_map.values()]

    df.to_csv(os.path.join(cache_folder(), f'TRIALS.csv'), index=False)

def load_from_csv(filename: str):
    return pd.read_csv(os.path.join(cache_folder(), f'{filename}.csv'))

def load_e1_from_csv():
    return load_from_csv(foldername_1)

def load_e2_from_csv():
    return load_from_csv(foldername_2)

def load_e3_from_csv():
    return load_from_csv(foldername_3)

def load_all_from_csv():
    df1, df2, df3 = load_e1_from_csv(), load_e2_from_csv(), load_e3_from_csv()
    return pd.concat([df1, df2, df3])


if __name__ == "__main__":
    # for foldername in [foldername_1, foldername_2, foldername_3]:
    #     fix_response_in_data_files(foldername)

    # # list unique session names
    # for foldername in [foldername_1, foldername_2, foldername_3]:
    #     session_names = unique_session_names(foldername)
    #     for session_name in session_names:
    #         print(session_name)
    #     print('*******************************************')

    # # save session names ordered by date to a file
    # with open('session_names_ordered_by_datetime.txt', 'w') as f:
    #     f.write('\t'.join([
    #         'foldername',
    #         'participant_name',
    #         'session_date',
    #         'session_time',
    #         'session_duration',
    #         'session_name',
    #         'session_trials',
    #         'session_hits']) + '\n')
    #     for foldername in [foldername_1, foldername_2, foldername_3]:
    #         for participant in inspect_session_names_ordered_by_date(foldername):
    #             if participant[0]['participant_name'] in participants_to_ignore:
    #                 continue
    #             for session in participant:
    #                 trials = [trial['result'] for trial in session['session']]
    #                 f.write('\t'.join([
    #                     foldername,
    #                     session['participant_name'],
    #                     session['session_date_time'],
    #                     session['session_duration'].to_string_minutes(),
    #                     session['session_name'],
    #                     str(len(trials)),
    #                     str(sum(trials))]) + '\n')

    # # list duplicates/missing/total sessions by participant to a file
    # with open('totals_by_participant.txt', 'w') as f:
    #     for foldername in [foldername_1, foldername_2, foldername_3]:
    #         total_time = 0
    #         for participant in inspect_total_sessions_by_participant(foldername):
    #             if participant['participant_name'] in participants_to_ignore:
    #                 continue
    #             total_time += participant['total_duration']
    #             f.write('\t'.join([
    #                         foldername,
    #                         participant['participant_name'],
    #                         str(participant['sessions']),
    #                         participant['total_duration'].to_string_hours() + '\n']))

    #     f.write('\t'.join([f'Total time: {total_time.to_string_hours()}']))

    # for foldername in [foldername_1, foldername_2, foldername_3]:
    #     save_participant_sources(foldername, True)

    export_to_csv()
    # export_to_csv_2()
    # export_to_csv(foldername_3)
    # export_to_csv_all()
