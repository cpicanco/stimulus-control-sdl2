from fileutils import as_data, file_exists, directory_exists
from fileutils import cd, list_files, list_data_folders, load_file
from fileutils import walk_and_execute
from classes import Information
from metadata import Metadata
from datetime import timedelta
from barplots import bar_plot, bar_subplots, box_plot, dispersion_plot_per_cycle, barplot_per_cycle
from barplots import barplot_per_participant
from lineplots import line_plot_per_cycle
from anonimizator import anonimize, deanonimize
from constants import participants_natural
from constants import participants_social

import pandas as pd

def calculate_measures_in_cycle_and_save_to_metadata(folder, date, cycle):
    def collect_data():
        time1 = timedelta()
        time2 = timedelta()

        container = []
        levenstein = {
            'CD_Training': [],
            'CD_Probes_1': [],
            'CD_Probes_2': [],
            'CD_Probes_3': []}

        probe_file_count = 0
        for entry in list_files('.info.processed'):
            data_file = as_data(entry, processed=True)
            probe_file = data_file.replace('.data', '.probes')
            if not file_exists(data_file):
                continue

            info = Information(entry)

            if info.has_valid_result():
                container.append(load_file(data_file))
                if 'Pre-treino' in info.session_name:
                    time1 += info.duration.value
                else:
                    time2 += info.duration.value

                if file_exists(probe_file):
                    if 'Treino-AC-CD' in info.session_name:
                        levenstein['CD_Training'].append(load_file(probe_file)['Levenshtein'])

                    elif 'Sondas-CD-Palavras-12-ensino-8-generalizacao' in info.session_name:
                        probe_file_count += 1
                        if probe_file_count == 1:
                            levenstein['CD_Probes_1'].append(load_file(probe_file)['Levenshtein'])
                        elif probe_file_count == 2:
                            levenstein['CD_Probes_3'].append(load_file(probe_file)['Levenshtein'])
                        else:
                            raise ValueError(f'Unexpected probe count {probe_file_count} in file:'+probe_file)

                    elif 'Sondas-CD-Palavras-generalizacao-reservadas' in info.session_name:
                        levenstein['CD_Probes_2'].append(load_file(probe_file)['Levenshtein'])

        if len(container) == 0:
            return None, None, None
        else:
            return pd.concat(container), levenstein, (time1, time2)

    participant = anonimize(folder, as_path=False).split('-')[1]
    participant_code = anonimize(folder)
    data, levenstein, time = collect_data()
    if data is None:
        return

    cycle = cycle
    total_pre_training_duration, total_training_probes_duration = time
    total_pre_training_trials = 0
    total_pre_training_bb_trials = 0
    total_pre_training_cd_trials = 0

    total_training_trials = 0
    total_training_hits = 0
    total_training_misses = 0

    total_teaching_trials = 0
    total_teaching_hits = 0
    total_teaching_misses = 0

    total_AB_training_trials = 0
    total_AB_training_hits = 0
    total_AB_training_misses = 0

    total_AC_training_trials = 0
    total_AC_training_hits = 0
    total_AC_training_misses = 0

    total_CD_training_trials = 0
    total_CD_training_hits = 0
    total_CD_training_misses = 0

    total_probe_trials = 0
    total_probe_hits = 0
    total_probe_misses = 0

    total_BC_w1_probe_trials = 0
    total_BC_w1_probe_hits = 0
    total_BC_w1_probe_misses = 0

    total_CB_w1_probe_trials = 0
    total_CB_w1_probe_hits = 0
    total_CB_w1_probe_misses = 0

    total_BC_w2_probe_trials = 0
    total_BC_w2_probe_hits = 0
    total_BC_w2_probe_misses = 0

    total_CB_w2_probe_trials = 0
    total_CB_w2_probe_hits = 0
    total_CB_w2_probe_misses = 0

    total_CD_w1_probe_trials = 0
    total_CD_w1_probe_hits = 0
    total_CD_w1_probe_misses = 0

    total_CD_w2_probe_trials = 0
    total_CD_w2_probe_hits = 0
    total_CD_w2_probe_misses = 0

    total_AC_probe_trials = 0
    total_AC_probe_hits = 0
    total_AC_probe_misses = 0

    try:
        levenstein['CD_Training'] = pd.concat(levenstein['CD_Training'])
    except ValueError:
        levenstein['CD_Training'] = pd.DataFrame()
        levenstein['CD_Training']['Levenshtein'] = [0]

    try:
        levenstein['CD_Probes_1'] = pd.concat(levenstein['CD_Probes_1'])
    except ValueError:
        levenstein['CD_Probes_1'] = pd.DataFrame()
        levenstein['CD_Probes_1']['Levenshtein'] = [0]

    try:
        levenstein['CD_Probes_2'] = pd.concat(levenstein['CD_Probes_2'])
    except ValueError:
        levenstein['CD_Probes_2'] = pd.DataFrame()
        levenstein['CD_Probes_2']['Levenshtein'] = [0]

    try:
        levenstein['CD_Probes_3'] = pd.concat(levenstein['CD_Probes_3'])
    except ValueError:
        levenstein['CD_Probes_3'] = pd.DataFrame()
        levenstein['CD_Probes_3']['Levenshtein'] = [0]

    pre_training_trials = data[data['Condition'] == 0]
    total_pre_training_trials = pre_training_trials.shape[0]

    pre_training_bb_trials = pre_training_trials[pre_training_trials['Relation'] == 'B-B']
    total_pre_training_bb_trials = pre_training_bb_trials.shape[0]

    pre_training_cd_trials = pre_training_trials[pre_training_trials['Relation'] == 'C-D']
    total_pre_training_cd_trials = pre_training_cd_trials.shape[0]

    data = data[data['Condition'] != 0]

    total_training_trials = data.shape[0]

    # hits
    hits = data[data['Result'] == 'Hit']
    total_training_hits = hits.shape[0]

    misses = data[data['Result'] == 'Miss']
    total_training_misses = misses.shape[0]

    # teaching trials
    training_trials = data[data['HasDifferentialReinforcement'] == True]
    total_teaching_trials = training_trials.shape[0]

    teaching_hits = training_trials[training_trials['Result'] == 'Hit']
    total_teaching_hits = teaching_hits.shape[0]

    teaching_misses = training_trials[training_trials['Result'] == 'Miss']
    total_teaching_misses = teaching_misses.shape[0]

    AB_training = training_trials[training_trials['Relation'] == 'A-B']
    total_AB_training_trials = AB_training.shape[0]

    AB_training_hits = AB_training[AB_training['Result'] == 'Hit']
    total_AB_training_hits = AB_training_hits.shape[0]

    AB_training_misses = AB_training[AB_training['Result'] == 'Miss']
    total_AB_training_misses = AB_training_misses.shape[0]

    AC_training = training_trials[training_trials['Relation'] == 'A-C']
    total_AC_training_trials = AC_training.shape[0]

    AC_training_hits = AC_training[AC_training['Result'] == 'Hit']
    total_AC_training_hits = AC_training_hits.shape[0]

    AC_training_misses = AC_training[AC_training['Result'] == 'Miss']
    total_AC_training_misses = AC_training_misses.shape[0]

    CD_training = training_trials[training_trials['Relation'] == 'C-D']
    total_CD_training_trials = CD_training.shape[0]

    CD_training_hits = CD_training[CD_training['Result'] == 'Hit']
    total_CD_training_hits = CD_training_hits.shape[0]

    CD_training_misses = CD_training[CD_training['Result'] == 'Miss']
    total_CD_training_misses = CD_training_misses.shape[0]

    # probe trials
    probe_trials = data[data['HasDifferentialReinforcement'] == False]
    total_probe_trials = probe_trials.shape[0]

    probe_hits = probe_trials[probe_trials['Result'] == 'Hit']
    total_probe_hits = probe_hits.shape[0]

    probe_misses = probe_trials[probe_trials['Result'] == 'Miss']
    total_probe_misses = probe_misses.shape[0]

    BC_w1_probes = data[(data['Relation'] == 'B-C') & (data['Condition'] == 3)]
    total_BC_w1_probe_trials = BC_w1_probes.shape[0]

    BC_w1_hits = BC_w1_probes[BC_w1_probes['Result'] == 'Hit']
    total_BC_w1_probe_hits = BC_w1_hits.shape[0]

    BC_w1_misses = BC_w1_probes[BC_w1_probes['Result'] == 'Miss']
    total_BC_w1_probe_misses = BC_w1_misses.shape[0]

    CB_w1_probes = data[(data['Relation'] == 'C-B') & (data['Condition'] == 3)]
    total_CB_w1_probe_trials = CB_w1_probes.shape[0]

    CB_w1_hits = CB_w1_probes[CB_w1_probes['Result'] == 'Hit']
    total_CB_w1_probe_hits = CB_w1_hits.shape[0]

    CB_w1_misses = CB_w1_probes[CB_w1_probes['Result'] == 'Miss']
    total_CB_w1_probe_misses = CB_w1_misses.shape[0]

    BC_w2_probes = data[(data['Relation'] == 'B-C') & (data['Condition'] == 4)]
    total_BC_w2_probe_trials = BC_w2_probes.shape[0]

    BC_w2_hits = BC_w2_probes[BC_w2_probes['Result'] == 'Hit']
    total_BC_w2_probe_hits = BC_w2_hits.shape[0]

    BC_w2_misses = BC_w2_probes[BC_w2_probes['Result'] == 'Miss']
    total_BC_w2_probe_misses = BC_w2_misses.shape[0]

    CB_w2_probes = data[(data['Relation'] == 'C-B') & (data['Condition'] == 4)]
    total_CB_w2_probe_trials = CB_w2_probes.shape[0]

    CB_w2_hits = CB_w2_probes[CB_w2_probes['Result'] == 'Hit']
    total_CB_w2_probe_hits = CB_w2_hits.shape[0]

    CB_w2_misses = CB_w2_probes[CB_w2_probes['Result'] == 'Miss']
    total_CB_w2_probe_misses = CB_w2_misses.shape[0]

    CD_w2_probes = data[(data['Relation'] == 'C-D') & (data['Condition'] == 5)]
    total_CD_w2_probe_trials = CD_w2_probes.shape[0]

    CD_w2_hits = CD_w2_probes[CD_w2_probes['Result'] == 'Hit']
    total_CD_w2_probe_hits = CD_w2_hits.shape[0]

    CD_w2_misses = CD_w2_probes[CD_w2_probes['Result'] == 'Miss']
    total_CD_w2_probe_misses = CD_w2_misses.shape[0]

    mean_CD_w2_leveshtein = levenstein['CD_Probes_2'].values.mean()

    CD_probes = data[(data['Relation'] == 'C-D') & (data['Condition'] == 7)]

    # get unique times
    times = CD_probes['Time'].unique()
    # get the first time for CD_w1_probes
    if len(times) == 0:
        total_CD_w1_probe_trials = 0
        total_CD_w1_probe_hits = None
        total_CD_w1_probe_misses = None
        mean_CD_w1_leveshtein = None

        total_CD_w3_probe_trials = 0
        total_CD_w3_probe_hits = None
        total_CD_w3_probe_misses = None
        mean_CD_w3_leveshtein = None

    elif len(times) == 1:
        time = times[0]
        CD_w1_probes = CD_probes[CD_probes['Time'] == time]
        total_CD_w1_probe_trials = CD_w1_probes.shape[0]

        CD_w1_hits = CD_w1_probes[CD_w1_probes['Result'] == 'Hit']
        total_CD_w1_probe_hits = CD_w1_hits.shape[0]

        CD_w1_misses = CD_w1_probes[CD_w1_probes['Result'] == 'Miss']
        total_CD_w1_probe_misses = CD_w1_misses.shape[0]

        mean_CD_w1_leveshtein = levenstein['CD_Probes_1'].values.mean()

        total_CD_w3_probe_trials = 0
        total_CD_w3_probe_hits = None
        total_CD_w3_probe_misses = None
        mean_CD_w3_leveshtein = None

        # override values for participant 12-MED on 2024-04-06
        if folder == '12-MED':
            if date == '2024-04-06':
                total_CD_w1_probe_trials = 0
                total_CD_w1_probe_hits = None
                total_CD_w1_probe_misses = None
                mean_CD_w1_leveshtein = None

                CD_w3_probes = CD_probes[CD_probes['Time'] == time]
                total_CD_w3_probe_trials = CD_w3_probes.shape[0]

                CD_w3_hits = CD_w3_probes[CD_w3_probes['Result'] == 'Hit']
                total_CD_w3_probe_hits = CD_w3_hits.shape[0]

                CD_w3_misses = CD_w3_probes[CD_w3_probes['Result'] == 'Miss']
                total_CD_w3_probe_misses = CD_w3_misses.shape[0]

                mean_CD_w3_leveshtein = levenstein['CD_Probes_3'].values.mean()

    elif len(times) == 2:
        time = times[0]
        CD_w1_probes = CD_probes[CD_probes['Time'] == time]

        total_CD_w1_probe_trials = CD_w1_probes.shape[0]

        CD_w1_hits = CD_w1_probes[CD_w1_probes['Result'] == 'Hit']
        total_CD_w1_probe_hits = CD_w1_hits.shape[0]

        CD_w1_misses = CD_w1_probes[CD_w1_probes['Result'] == 'Miss']
        total_CD_w1_probe_misses = CD_w1_misses.shape[0]

        mean_CD_w1_leveshtein = levenstein['CD_Probes_1'].values.mean()

        time = times[1]
        CD_w3_probes = CD_probes[CD_probes['Time'] == time]

        total_CD_w3_probe_trials = CD_w3_probes.shape[0]

        CD_w3_hits = CD_w3_probes[CD_w3_probes['Result'] == 'Hit']
        total_CD_w3_probe_hits = CD_w3_hits.shape[0]

        CD_w3_misses = CD_w3_probes[CD_w3_probes['Result'] == 'Miss']
        total_CD_w3_probe_misses = CD_w3_misses.shape[0]

        mean_CD_w3_leveshtein = levenstein['CD_Probes_3'].values.mean()
    else:
        raise ValueError("Unexpected number of times for CD probes.")

    AC_probes = data[(data['Relation'] == 'A-C') & (data['Condition'] == 6)]
    total_AC_probe_trials = AC_probes.shape[0]

    AC_hits = AC_probes[AC_probes['Result'] == 'Hit']
    total_AC_probe_hits = AC_hits.shape[0]

    AC_misses = AC_probes[AC_probes['Result'] == 'Miss']
    total_AC_probe_misses = AC_misses.shape[0]

    metadata = Metadata()
    metadata.items.clear()
    metadata_dict = {
        'participant': participant,
        'participant_code': participant_code,
        'date': date,
        'cycle': cycle,
        'total_pre_training_duration': str(total_pre_training_duration),
        'total_pre_training_trials': str(total_pre_training_trials),
        'total_pre_training_bb_trials': str(total_pre_training_bb_trials),
        'total_pre_training_cd_trials': str(total_pre_training_cd_trials),
        'total_training_probes_duration': str(total_training_probes_duration),
        'total_training_trials': str(total_training_trials),
        'total_training_hits': str(total_training_hits),
        'total_training_misses': str(total_training_misses),
        'total_teaching_trials': str(total_teaching_trials),
        'total_teaching_hits': str(total_teaching_hits),
        'total_teaching_misses': str(total_teaching_misses),
        'total_probe_trials': str(total_probe_trials),
        'total_probe_hits': str(total_probe_hits),
        'total_probe_misses': str(total_probe_misses),
        'total_AB_training_trials': str(total_AB_training_trials),
        'total_AB_training_hits': str(total_AB_training_hits),
        'total_AB_training_misses': str(total_AB_training_misses),
        'total_AC_training_trials': str(total_AC_training_trials),
        'total_AC_training_hits': str(total_AC_training_hits),
        'total_AC_training_misses': str(total_AC_training_misses),
        'total_CD_training_trials': str(total_CD_training_trials),
        'total_CD_training_hits': str(total_CD_training_hits),
        'total_CD_training_misses': str(total_CD_training_misses),
        'total_BC_w1_probe_trials': str(total_BC_w1_probe_trials),
        'total_BC_w1_probe_hits': str(total_BC_w1_probe_hits),
        'total_BC_w1_probe_misses': str(total_BC_w1_probe_misses),
        'total_CB_w1_probe_trials': str(total_CB_w1_probe_trials),
        'total_CB_w1_probe_hits': str(total_CB_w1_probe_hits),
        'total_CB_w1_probe_misses': str(total_CB_w1_probe_misses),
        'total_BC_w2_probe_trials': str(total_BC_w2_probe_trials),
        'total_BC_w2_probe_hits': str(total_BC_w2_probe_hits),
        'total_BC_w2_probe_misses': str(total_BC_w2_probe_misses),
        'total_CB_w2_probe_trials': str(total_CB_w2_probe_trials),
        'total_CB_w2_probe_hits': str(total_CB_w2_probe_hits),
        'total_CB_w2_probe_misses': str(total_CB_w2_probe_misses),
        'total_CD_w1_probe_trials': str(total_CD_w1_probe_trials),
        'total_CD_w1_probe_hits': str(total_CD_w1_probe_hits),
        'total_CD_w1_probe_misses': str(total_CD_w1_probe_misses),
        'mean_CD_w1_leveshtein': str(mean_CD_w1_leveshtein),
        'total_CD_w2_probe_trials': str(total_CD_w2_probe_trials),
        'total_CD_w2_probe_hits': str(total_CD_w2_probe_hits),
        'total_CD_w2_probe_misses': str(total_CD_w2_probe_misses),
        'mean_CD_w2_leveshtein': str(mean_CD_w2_leveshtein),
        'total_CD_w3_probe_trials': str(total_CD_w3_probe_trials),
        'total_CD_w3_probe_hits': str(total_CD_w3_probe_hits),
        'total_CD_w3_probe_misses': str(total_CD_w3_probe_misses),
        'mean_CD_w3_leveshtein': str(mean_CD_w3_leveshtein),
        'total_AC_probe_trials': str(total_AC_probe_trials),
        'total_AC_probe_hits': str(total_AC_probe_hits),
        'total_AC_probe_misses': str(total_AC_probe_misses)
    }
    metadata.items.update(metadata_dict)
    metadata.save()

def create_metadata():
    cd('..')
    participant_folders = list_data_folders()
    for participant_folder in participant_folders:
        cd(participant_folder)
        cd('analysis')

        safety_copy_folders_by_date = list_data_folders()
        for date_folder in safety_copy_folders_by_date:
            cd(date_folder)
            safety_copy_folders_by_cycle = list_data_folders()
            for cycle_folder in safety_copy_folders_by_cycle:
                cd(cycle_folder)
                calculate_measures_in_cycle_and_save_to_metadata(participant_folder, date_folder, cycle_folder)
                cd('..')
            cd('..')

        cd('..')
        cd('..')

def calculate_hit_rate(data, key_hits, key_trials, separate_denominators):
    try:
        if separate_denominators:
            return (int(data[key_hits]), int(data[key_trials]))
        else:
            return int(data[key_hits]) / int(data[key_trials])
    except ZeroDivisionError:
        return None

def collect_metadata(container,
                     plot_individual_days=False,
                     use_levenstein=False,
                     separate_denominators=False):
    data = Metadata().items
    try:
        participant = data['participant']
    except KeyError:
        return

    date = data['date']
    cycle = data['cycle']

    total_training_probes_duration = data['total_training_probes_duration']
    if total_training_probes_duration == '0':
        return

    cd_probes_1_hit_rate = calculate_hit_rate(data, 'total_CD_w1_probe_hits', 'total_CD_w1_probe_trials', separate_denominators)

    cd_probes_2_hit_rate = calculate_hit_rate(data, 'total_CD_w2_probe_hits', 'total_CD_w2_probe_trials', separate_denominators)

    cd_probes_3_hit_rate = calculate_hit_rate(data, 'total_CD_w3_probe_hits', 'total_CD_w3_probe_trials', separate_denominators)

    bc_probes_1_hit_rate = calculate_hit_rate(data, 'total_BC_w1_probe_hits', 'total_BC_w1_probe_trials', separate_denominators)

    cb_probes_1_hit_rate = calculate_hit_rate(data, 'total_CB_w1_probe_hits', 'total_CB_w1_probe_trials', separate_denominators)

    bc_probes_2_hit_rate = calculate_hit_rate(data, 'total_BC_w2_probe_hits', 'total_BC_w2_probe_trials', separate_denominators)

    cb_probes_2_hit_rate = calculate_hit_rate(data, 'total_CB_w2_probe_hits', 'total_CB_w2_probe_trials', separate_denominators)

    ac_probes_hit_rate = calculate_hit_rate(data, 'total_AC_probe_hits', 'total_AC_probe_trials', separate_denominators)

    ab_training_hit_rate = calculate_hit_rate(data, 'total_AB_training_hits', 'total_AB_training_trials', separate_denominators)

    ac_training_hit_rate = calculate_hit_rate(data, 'total_AC_training_hits', 'total_AC_training_trials', separate_denominators)

    cd_training_hit_rate = calculate_hit_rate(data, 'total_CD_training_hits', 'total_CD_training_trials', separate_denominators)

    identification = [
        ("Participant", participant),
        ("Date", date),
        ("Total Cycle Duration", total_training_probes_duration),
        ("Cycle", cycle)
    ]

    categories = [
        ("CD Probes 1", cd_probes_1_hit_rate, 'red'),
        ("AB Training", ab_training_hit_rate, 'blue'),
        ("AC Training", ac_training_hit_rate, 'blue'),
        ("CD Training", cd_training_hit_rate, 'blue'),
        ("BC Probes 1", bc_probes_1_hit_rate, 'green'),
        ("CB Probes 1", cb_probes_1_hit_rate, 'green'),
        ("BC Probes 2", bc_probes_2_hit_rate, 'yellow'),
        ("CB Probes 2", cb_probes_2_hit_rate, 'yellow'),
        ("CD Probes 2", cd_probes_2_hit_rate, 'red'),
        ("AC Probes", ac_probes_hit_rate, 'purple'),
        ("CD Probes 3", cd_probes_3_hit_rate, 'red')
    ]

    if plot_individual_days:
        bar_plot(categories, identification)

    container.append({'categories':categories, 'identification':identification})


def single_participant_plots():
    names = ['AB Training', 'AC Training', 'CD Training']
    names = names + ['BC Probes 1', 'CB Probes 1', 'BC Probes 2', 'CB Probes 2']
    names = names + ['CD Probes 1', 'CD Probes 2']
    names = names + ['AC Probes']

    # get the first two characters of each name and remove duplicates
    codes = list(set([name[:2] for name in names]))
    # sort the codes
    codes.sort()
    # join codes with _ and append to filename
    codes = '_'.join(codes)

    cd('..')
    participant_folders = list_data_folders()
    for folder in participant_folders:
        container = []
        walk_and_execute(folder, collect_metadata, container, False, True)
        participant = anonimize(folder, as_path=False).split('-')[1]
        # bar_subplots(container, True)
        # barplot_per_cycle(container=container, save=True, include_names=names, append_to_filename='_'+codes+'_'+participant+'_barplot')
        # dispersion_plot_per_cycle(container, True, style='scatter', include_names=names, append_to_filename='_'+codes+'_'+participant)

def plot(include_list=[], append_to_filename='', group=False):
    # you asked for ...
    print("You asked a group analysis for participants:", include_list)

    cd('..')
    container = []
    participant_folders = list_data_folders(include_list=include_list)
    for folder in participant_folders:
        walk_and_execute(folder, collect_metadata, container, False, False)
    # i delivered ... collect all participants included in the container
    participants = list(set([data['identification'][0][1] for data in container]))
    print("Doing for participants:", participants)

    cd('analysis')
    cd('output')

    # names = ['BC Probes 1', 'CB Probes 1', 'BC Probes 2', 'CB Probes 2']
    # box_plot(container, True, include_names=names)
    # dispersion_plot_per_cycle(container, False, include_names=names)

    names = ['BC Probes 2', 'CB Probes 2', 'AC Probes', 'CD Probes 2']
    line_plot_per_cycle(container, True,
                        include_names=names,
                        append_to_filename='_hit_rate'+append_to_filename,
                        group=group)
    cd('..')

def group_barplot(include_list=[]):
    # you asked for ...
    print("You asked a group analysis for participants:", include_list)

    cd('..')
    container = []
    participant_folders = list_data_folders(include_list=include_list)
    for folder in participant_folders:
        walk_and_execute(folder, collect_metadata, container, False, False)

    # I delivered ... collect all participants included in the container
    participants = list(set([data['identification'][0][1] for data in container]))
    print("Doing for participants:", participants)

    names = ['CD Probes 3']
    barplot_per_participant(container, save=True, include_names=names, append_to_filename='_group_barplot')


def fix_cycles():
    cd('..')
    participant_folders = list_data_folders()
    for participant_folder in participant_folders:
        cd(participant_folder)
        # check if directory analysis exists
        if directory_exists('analysis'):
            cd('analysis')

            safety_copy_folders_by_date = list_data_folders()
            for date_folder in safety_copy_folders_by_date:
                cd(date_folder)

                safety_copy_folders_by_cycle = list_data_folders()
                for cycle in safety_copy_folders_by_cycle:
                    cd(cycle)
                    for entry in list_files('.info.processed'):
                        info = Information(entry)
                        if info.has_valid_result():
                            data_file = as_data(entry, processed=True)
                            data = pd.read_csv(data_file, sep='\t')
                            data['Cycle.ID'] = cycle
                            data.to_csv(data_file, sep='\t', index=False)
                            print(f"Cycle fixed in {data_file}.")
                    cd('..')
                cd('..')
            cd('..')
        else:
            print(f"Directory 'analysis' not found in {participant_folder}. Skipping...")
        cd('..')

if __name__ == "__main__":
    # fix_cycles()
    # single_participant_plots()

    from constants import participants_natural, participants_social
    # pilot study
    # participants_none = [
    # deanonimize('1-SAR'),
    # deanonimize('2-FIO'),
    # deanonimize('4-LIV'),
    # deanonimize('5-JUL'),
    # deanonimize('6-MAR'),
    # '9-CES',
    # ]

    plot(participants_natural, append_to_filename='_natural', group=True)
    plot(participants_social, append_to_filename='_social', group=True)
    plot(participants_natural + participants_social, append_to_filename='_all', group=True)

    participants = participants_natural
    for participant in participants:
        plot(participant, append_to_filename='_natural_' + participant.replace('-', '_'))

    participants = participants_social
    for participant in participants:
        plot(participant, append_to_filename='_social_' + participant.replace('-', '_'))