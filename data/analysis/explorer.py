import os
import pickle

from words import words_per_file

from classes import (
    Information
)

from fileutils import (
    cd,
    design_pseudowords_dirname,
    data_dirname,
    list_data_folders,
    list_files,
    cache_folder
)

def save_pickle(filename, data):
    folder = cache_folder()
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = os.path.join(folder, filename +'.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def participant_sources():
    from player_synchronizer import Synchronizer

    cd(design_pseudowords_dirname())
    design_files = [os.path.splitext(f)[0] for f in os.listdir('.') if os.path.isfile(f)]
    design_files.sort()
    cd(data_dirname())
    sources = {}
    for participant in list_data_folders(directory_name=data_dirname()):

        if participant == '2-FEB' \
           or participant == '6-RIC' \
           or participant == '11-VSC' \
           or participant == '18-JEG' \
           or participant == '21-NLC' \
           or participant == '24-VCC':
            print('This participant has incomplete data. Skipping...')
            continue

        sources[participant] = {}
        cd(os.path.join(participant, 'analysis'))
        for date_folder in list_data_folders():
            cd(date_folder)
            for cycle_folder in list_data_folders():
                cd(cycle_folder)
                for entry in list_files('.info.processed'):
                    info = Information(entry)
                    if info.has_valid_result():
                        for design_file in design_files:
                            if info.session_name == design_file:
                                # if key does not exist, create it
                                if sources[participant].get(design_file) is None:
                                    sources[participant][design_file] = []

                                if participant == '3-PVV' \
                                   and cycle_folder == '5' \
                                   and date_folder == '2024-07-25' \
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
                                    'date': date_folder,
                                    'cycle': cycle_folder,
                                    'path': os.path.join(participant, 'analysis', date_folder, cycle_folder),
                                    'code': code,
                                    'session' : Synchronizer(code)
                                })
                cd('..')
            cd('..')
        cd('..')
        cd('..')

    return sources

def save_participant_sources():
    sources = participant_sources()
    for source in sources:
        # count how many design files are for this participant, we need 50
        if len(sources[source]) < 50:
            print(f'Warning: {source} has only {len(sources[source])} design files')

        for design_file in sources[source]:
            number = len(sources[source][design_file])
            if number == 0:
                print(f'Warning: {source} has no data for {design_file}')
            elif number > 1:
                print(f'Warning: {source} has multiple data (n={number}) for {design_file}')
                for data in sources[source][design_file]:
                    print(f'  - {data["date"]}, {data["cycle"]}, {data["code"]}')

    save_pickle('sources', sources)

def load_pickle(filename):
    output_path = os.path.join(cache_folder(), filename +'.pkl')
    with open(output_path, 'rb') as f:
        return pickle.load(f)

def load_participant_sources() -> dict:
    # load sources
    output_path = os.path.join(cache_folder(), 'sources.pkl')
    if not os.path.exists(output_path):
        print(f"Cache file '{output_path}' not found.")
        return {}

    with open(output_path, 'rb') as f:
        return pickle.load(f)

def save_participant_objects(participant_objects: dict):
    pass

def load_participant_objects(participant: str):
    pass

def load_participants() -> dict:
    pass

if __name__ == '__main__':
    # save_participant_sources()

    sources = load_participant_sources()
    for participant in sources:
        target_sources = []
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    session = source['session']
                    print(session.duration)
                    # session_filtered = session.word_filter('')
                    # source['trials'] = session_filtered.trials
                    # for trial in source['trials']:
                    #     print(trial['duration'])