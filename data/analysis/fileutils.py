import os
import time
import datetime
import shutil

import pandas as pd

from anonimizator import anonimize
from headers import info_header

data_folder_name = 'data'
design_folder_name = 'design'
project_folder_name = 'pseudowords'

def cache_folder():
    return os.path.join(data_dirname(), 'analysis', 'cache')

def design_pseudowords_dirname():
    # get script path
    script_path = os.path.dirname(os.path.realpath(__file__))
    # go up two folders
    script_path = os.path.dirname(script_path)
    script_path = os.path.dirname(script_path)
    return os.path.join(script_path, design_folder_name, project_folder_name)

def data_dirname():
    script_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.dirname(script_path)
    script_path = os.path.dirname(script_path)
    return os.path.join(script_path, data_folder_name)

def return_to(directory, verbose=True):
    # check if current directory is data
    if os.getcwd().endswith(directory):
        return
    else:
        # recursively return until reach data directory
        cd('..', verbose)
        data_dir(verbose)

def file_exists(entry):
    return os.path.exists(entry)

def data_dir(verbose=True):
    return_to(data_folder_name, verbose)

def directory_exists(directory):
    return os.path.isdir(directory)

def cd(directory, verbose=True):
    os.chdir(directory)
    if verbose:
        print("Current Working Directory: ", os.getcwd())

def get_real_filepath(entry):
    return os.path.join(os.getcwd(), entry)

def load_file(entry):
    df = None
    if entry.endswith('.info.processed'):
        df = pd.read_csv(entry, sep='\t', encoding='utf-8', header=None, index_col=0, engine='python')

    elif entry.endswith('.probes.processed'):
        df = pd.read_csv(entry, sep='\t')

    elif entry.endswith('.data.processed'):
        df = pd.read_csv(entry, sep='\t', header=0, engine='python')
        if 'Cycle.ID' in df.columns:
            df['Cycle.ID'] = pd.to_numeric(df['Cycle.ID'], errors='coerce').fillna(0).astype(int)

    else:
        df = pd.read_csv(entry, sep='\t', header=0, engine='python')

    return df

def folder_sorter(x):
    return int(x.split('-')[0])

def list_data_folders(include_list=[], exclude_list=[], directory_name='.'):
    all_entries = os.listdir(directory_name)
    if len(include_list) == 0:
        exclude_list += [
            '.vscode',
            '__pycache__',
            'analysis',
            'dados-brasil',
            'dados-brasil-geovana',
            'dados-portugal',
            'dados-portugal-geovana',
            'output',
            '0-Rafael',
            '3-Teste',
            '7-Teste2']

        entries = [entry for entry in all_entries \
                if os.path.isdir(os.path.join(directory_name, entry)) \
                and entry not in exclude_list]
        entries.sort(key=folder_sorter)
        return entries
    else:
        entries = [entry for entry in all_entries \
                if os.path.isdir(os.path.join(directory_name, entry)) \
                and entry in include_list].sort(key=folder_sorter)
        entries.sort(key=folder_sorter)
        return entries

def get_data_folders(anonimized=False):
    data_dir()
    participant_folders = list_data_folders()
    if anonimized:
        participant_folders = [anonimize(folder) for folder in participant_folders]
    participant_folders.sort(key=folder_sorter)
    return participant_folders

def list_files(extension='', except_name=''):
    def has_not_except_name(astring):
        if except_name:
            return except_name not in astring
        else:
            return True

    # Get all entries in the current directory except 'ID' and 'LastValidBaseFilename' files
    all_entries = os.listdir('.')
    # Filter out folders and files with different extensions
    return [entry for entry in all_entries \
               if os.path.isfile(entry) \
               and has_not_except_name(entry) \
               and entry != 'ID' \
               and entry != 'LastValidBaseFilename' \
               and entry.endswith(extension)]

def get_readable_creation_date(real_filepath):
    """
    Note: Windows only.
    """
    # Get the creation time of the file
    creation_time = os.path.getctime(real_filepath)
    # Convert the creation time to a human-readable format
    return time.ctime(creation_time)


def get_creation_date(real_filepath, format='%Y-%m-%d'):
    """
    Note: Windows only.
    """
    # Get the creation time of the file
    creation_time = os.path.getmtime(real_filepath)
    # Convert the creation time to a datetime object
    date_time_obj = datetime.datetime.fromtimestamp(creation_time)
    # Format the date as YYYY-MM-DD
    return date_time_obj.strftime(format)

def get_cycle(entry):
    info_filename = as_info(entry)
    with open(info_filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith(info_header[2]):
            session_name = line.split(':')[1].split('-')
            condition = int(session_name[1].replace('a', '').replace('b', ''))
            cycle = int(session_name[0].replace('Ciclo', ''))
            if condition == 7:
                if cycle == 6:
                    pass
                else:
                    cycle += 1
            else:
                pass
            return str(cycle)

def safety_copy(entry, cycle):
    source = get_real_filepath(entry)
    creation_date = get_creation_date(source)

    destination = os.path.join(os.getcwd(), 'analysis', creation_date, cycle, entry)

    # Extract the directory part of the destination path
    destination_dir = os.path.dirname(destination)

    # Check if the destination file already exists
    if not os.path.exists(destination):
        # Create intermediate directories if they don't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"File '{source}' copied to '{destination}'.")
    else:
        print(f"The file '{destination}' already exists.")

    return destination

def replace_extension(filename, new_extension, processed=False):
    if processed:
        filename = filename.replace('.processed', '')
    root, _ = os.path.splitext(filename)
    # Add the new extension
    return root + new_extension

def as_timestamps(entry, processed=False):
    if processed:
        return replace_extension(entry, '.timestamps.processed', processed=True)
    else:
        return replace_extension(entry, '.timestamps')

def as_data(entry, processed=False):
    if processed:
        return replace_extension(entry, '.data.processed', processed=True)
    else:
        return replace_extension(entry, '.data')

def as_info(entry, processed=False):
    if processed:
        return replace_extension(entry, '.info.processed', processed=True)
    else:
        return replace_extension(entry, '.info')

def walk_and_execute(entry, function, *args):
    cd(entry)
    try:
        cd('analysis')
    except FileNotFoundError:
        cd('..')
        return

    safety_copy_folders_by_date = list_data_folders()
    for date_folder in safety_copy_folders_by_date:
        cd(date_folder)

        safety_copy_folders_by_cycle = list_data_folders()
        for cycle_folder in safety_copy_folders_by_cycle:
            cd(cycle_folder)
            function(*args)
            cd('..')
        cd('..')
    cd('..')
    cd('..')

def delete_deprecated_files():
    def delete_probes_files():
        files = list_files('.probes.processed')
        if len(files) == 0:
            return
        else:
            for file in files:
                os.remove(file)
                print(f"File '{file}' deleted.")

    cd('..')
    participant_folders = list_data_folders()
    for folder in participant_folders:
        walk_and_execute(folder, delete_probes_files)

if __name__ == "__main__":
    delete_deprecated_files()