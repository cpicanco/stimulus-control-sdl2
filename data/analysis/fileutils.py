import os
import time
import datetime
import shutil

import pandas as pd

from metadata import Metadata

def file_exists(entry):
    return os.path.exists(entry)

def cd(directory):
    os.chdir(directory)
    print("Current Working Directory: ", os.getcwd())

def get_real_filepath(entry):
    return os.path.join(os.getcwd(), entry)

def load_file(entry):
    if entry.endswith('.info.processed'):
        return pd.read_csv(entry, sep='\t', encoding='utf-8', header=None, index_col=0, engine='python')

    if entry.endswith('.probes.processed'):
        return pd.read_csv(entry, sep='\t')

    return pd.read_csv(entry, sep='\t', header=0, engine='python')

def list_data_folders(include_list=[], exclude_list=[]):
    if len(include_list) == 0:
        exclude_list += ['__pycache__', 'analysis', '.vscode', 'output', '0-Rafael', '3-Teste', '7-Teste2']
        # Get all entries in the current directory
        all_entries = os.listdir('.')
        # Filter out files and excluded folders
        return [entry for entry in all_entries \
                if os.path.isdir(entry) \
                and entry not in exclude_list]
    else:
        all_entries = os.listdir('.')
        return [entry for entry in all_entries \
                if os.path.isdir(entry) \
                and entry in include_list]


def list_files(extension=''):
    # Get all entries in the current directory
    all_entries = os.listdir('.')
    # Filter out folders and files with different extensions
    return [entry for entry in all_entries \
               if os.path.isfile(entry) \
               and entry != 'ID' \
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

def safety_copy(entry):
    source = get_real_filepath(entry)
    creation_date = get_creation_date(source)
    destination = os.path.join(os.getcwd(), 'analysis', creation_date, entry)

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

def walk_and_execute_convertion(entry, function, *args):
    cd(entry)
    cd('analysis')

    metadata = Metadata()
    try:
        folder_done = bool(metadata.items['done'])
    except KeyError:
        folder_done = False

    if not folder_done:
        safety_copy_folders = list_data_folders()
        for data_folder in safety_copy_folders:
            cd(data_folder)
            function(*args)
            cd('..')
        metadata.items['done'] = str(True)
        metadata.save()

    cd('..')
    cd('..')

def walk_and_execute(entry, function, *args):
    cd(entry)
    cd('analysis')

    safety_copy_folders = list_data_folders()
    for data_folder in safety_copy_folders:
        cd(data_folder)
        function(*args)
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