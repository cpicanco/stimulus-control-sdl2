from fileutils import cd, walk_and_execute
from metadata import Metadata
from metadata_calculator import collect_metadata
from my_statistics import paired_chi_squared

from barplots import dispersion_plot_per_cycle
from barplots import default_axis_config, save_or_show

import numpy as np
import matplotlib.pyplot as plt

from words import nibo, fale, bofa, leni, lebo, fani, boni, lefa, fabo, nile, bole, nifa
from words import bona, lefi, fabe, nilo, lani, febo, nole, bifa


from speech import load_probes_file

def count_min_max(participants_list):
    cd('..')
    container = []
    for folder in participants_list:
        walk_and_execute(folder, collect_metadata, container, False, False, True)

    include_names = ["AB Training", "AC Training", "CD Training"]
    max_cycle = 0
    grouped_values = {}
    for data in container:
        identification = data['identification']
        categories = data['categories']
        # we need to group values per name then per cycle
        # cycle is the last element in identification tuple
        _, id_data = zip(*identification)
        max_cycle = max(max_cycle, int(id_data[3]))
        cycle = id_data[3]
        if cycle == '0':
            continue
        cycle = 'Cycle ' + cycle
        names, values, _ = zip(*categories)

        participant = id_data[0]
        if participant not in grouped_values:
            grouped_values[participant] = {cycle: {}}

        for name, value in zip(names, values):
            if name not in include_names:
                continue

            if value is None:
                continue

            if value[1] == 0:
                continue

            trials = value[1]

            if cycle in grouped_values[participant]:
                if name in grouped_values[participant][cycle]:
                    if isinstance(trials, (int, float)):
                        trials = [trials]
                    grouped_values[participant][cycle][name].extend(trials)
                else:
                    grouped_values[participant][cycle][name] = [trials]
            else:
                grouped_values[participant][cycle] = {name: [trials]}

    # a disctionary for each item in include_names
    teaching_trials = {name: [] for name in include_names}
    # calculate the min and max for each participant
    for participant in grouped_values:
        for cycle in grouped_values[participant]:
            for name in include_names:
                if name not in grouped_values[participant][cycle]:
                    continue
                value = grouped_values[participant][cycle][name]
                teaching_trials[name].extend(value)
                print(participant, cycle, name, value)

    for name in include_names:
        # show min, max, and median
        print(name, "min:", min(teaching_trials[name]), "max:", max(teaching_trials[name]), "median:", np.median(teaching_trials[name]))

def get_hits_trials_per_participant_per_cycle_per_phase(participants_list):
    cd('..')
    container = []
    for folder in participants_list:
        walk_and_execute(folder, collect_metadata, container, False, False, True)

    include_names = ["AB Training", "AC Training", "CD Training"]
    max_cycle = 0
    grouped_values = {}
    for data in container:
        identification = data['identification']
        categories = data['categories']
        # we need to group values per name then per cycle
        # cycle is the last element in identification tuple
        _, id_data = zip(*identification)
        max_cycle = max(max_cycle, int(id_data[3]))
        cycle = id_data[3]
        if cycle == '0':
            continue
        cycle = 'Cycle ' + cycle
        names, values, _ = zip(*categories)

        participant = id_data[0]
        if participant not in grouped_values:
            grouped_values[participant] = {cycle: {}}

        for name, value in zip(names, values):
            if name not in include_names:
                continue

            if value is None:
                continue

            if value[1] == 0:
                continue

            if len(value) > 2:
                raise ValueError("Unexpected value length:", value)

            value_list = [value[0], value[1]]
            if cycle in grouped_values[participant]:
                if name in grouped_values[participant][cycle]:
                    grouped_values[participant][cycle][name].append(value_list)
                else:
                    grouped_values[participant][cycle][name] = [value_list]
            else:
                grouped_values[participant][cycle] = {name: [value_list]}
    return grouped_values

def plot(participants_list=[]):
    # you asked for ...
    print("You asked a group analysis for participants:", participants_list)

    cd('..')
    container = []
    for folder in participants_list:
        walk_and_execute(folder, collect_metadata, container, False, False, True)

    # I delivered ... collect all participants included in the container
    participants = list(set([data['identification'][0][1] for data in container]))
    print("Doing for participants:", participants)

    # make a copy of the container
    container_for_dispersion_plot = container.copy()
    for data in container_for_dispersion_plot:
        for i, item in enumerate(data['categories']):
            if item[1] is not None:
                hits, trials = item[1]
                try:
                    data['categories'][i] = (item[0], trials-hits, item[2])
                except ZeroDivisionError:
                    print(data['identification'], item[0], item[1])
                    data['categories'][i] = (item[0], None, item[2])

    names = ["AB Training", "AC Training", "CD Training"]
    dispersion_plot_per_cycle(container_for_dispersion_plot,
                              save=True,
                              limit_y=False,
                              include_names=names,
                              style='violin',
                              append_to_filename='_group_dispersion_teaching')

def plot_rank_tests():
    """""
    data = {"teaching relation":[(U, p-value < .001), ...], ...}
    """
    data = {
        "AB vs AC": [(7385000.0, True), (7515000.0, True), (6630000.0, True), (5290000.0, True), (5785000.0, False), (5840000.0, False)],
        "AB vs CD": [(5435000.0, True), (6295000.0, True), (6050000.0, True), (4250000.0, True), (4135000.0, True), (4960000.0, True)],
        "AC vs CD": [(4825000.0, True), (4935000.0, True), (5295000.0, True), (4760000.0, True), (4465000.0, True), (4960000.0, True)]
    }

    # data to plot
    n_groups = 6

    AB_AC, AB_AC_p = zip(*data["AB vs AC"])
    AB_CD, AB_CD_p = zip(*data["AB vs CD"])
    AC_CD, AC_CD_p = zip(*data["AC vs CD"])

    fig, ax = plt.subplots()
    # start index at 1
    index = np.arange(1, n_groups + 1)
    # plot the results using lines
    ax.plot(index, AB_AC, label='AB vs AC', marker='o')
    ax.plot(index, AB_CD, label='AB vs CD', marker='o')
    ax.plot(index, AC_CD, label='AC vs CD', marker='o')
    ax.legend()
    plt.xlabel('Cycles')
    plt.ylabel('Rank (U)')

    plt.show()

def line_plot_AB_AC_CD():
    """
        Plot AB, AC, and CD Training trials (y) per cycle (x) for all participants.
    """
    # data to plot
    data = get_hits_trials_per_participant_per_cycle_per_phase(participants_list=participant_folders)

    # Initialize a dictionary to store the data for each training type
    data_to_plot = {}

    for participant in data:
        for cycle in data[participant]:
            if cycle not in data_to_plot:
                data_to_plot[cycle] = {}
            # we assume daily sessions, and we may have more than one day for each cycle
            # Participant "AND" is an exception with 2 days in cycle 1
            # so we sum trials here because we are interested in trials to criterion
            for name in data[participant][cycle]:
                # Initialize the list for this training type if it doesn't exist yet
                if name not in data_to_plot[cycle]:
                    data_to_plot[cycle][name] = []

                values = [value[1] for value in data[participant][cycle][name]]
                total = sum(values) if len(values) > 1 else values[0]
                data_to_plot[cycle][name].append(total)

    data_to_plot_for_real = {}
    for cycle in data_to_plot:
        for name in data_to_plot[cycle]:
            if name not in data_to_plot_for_real:
                data_to_plot_for_real[name] = []
            cycle_id = int(cycle.split()[1])
            data_to_plot_for_real[name].append((cycle_id, data_to_plot[cycle][name]))

    # save three plots, one for each training type, boxplot of trials per cycle, show mean=True

    for name in data_to_plot_for_real:
        fig, ax = plt.subplots()
        cycles, trials = zip(*data_to_plot_for_real[name])
        ax.boxplot(trials, positions=cycles, showmeans=True)
        plt.title(f'Trials to criterion along cycles for {name}')
        plt.xlabel('Cycles')
        plt.ylabel('Trials')

        # remove outlines of axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # before saving, group all data and plot a regression line
        x = []
        y = []
        for cycle, values in data_to_plot_for_real[name]:
            x.extend([cycle] * len(values))
            y.extend(values)

        # plot the regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", label=f"y={z[0]:.2f}x+{z[1]:.2f}")

        # make sure that min and max are included in the plot
        mininimun = min(y)
        maximum = max(y)
        if mininimun == 8:
            maximum = maximum + 1
        else:
            mininimun = mininimun - 10
            maximum = maximum + 10

        plt.ylim(mininimun, maximum)

        plt.legend()
        # save and close the plot
        plt.savefig(f'line_plot_{name}.pdf')
        plt.close()

def count_errors():
    data = get_hits_trials_per_participant_per_cycle_per_phase(participants_list=participant_folders)

    errors_per_participant_per_cycle_per_phase = {}
    for participant in data:
        if participant not in errors_per_participant_per_cycle_per_phase:
            errors_per_participant_per_cycle_per_phase[participant] = {}
        for cycle in data[participant]:
            if cycle not in errors_per_participant_per_cycle_per_phase[participant]:
                errors_per_participant_per_cycle_per_phase[participant][cycle] = {}
            for name in data[participant][cycle]:
                if name not in errors_per_participant_per_cycle_per_phase[participant][cycle]:
                    errors_per_participant_per_cycle_per_phase[participant][cycle][name] = []
                for value in data[participant][cycle][name]:
                    hits, trials = value
                    errors = trials - hits
                    errors_per_participant_per_cycle_per_phase[participant][cycle][name].append(errors)

    # now check if errors in AB are less than errors in AC
    for participant in errors_per_participant_per_cycle_per_phase:
        for cycle in errors_per_participant_per_cycle_per_phase[participant]:
            errors_AB = errors_per_participant_per_cycle_per_phase[participant][cycle]["AB Training"]
            errors_AC = errors_per_participant_per_cycle_per_phase[participant][cycle]["AC Training"]
            # errors_CD = errors_per_participant_per_cycle_per_phase[participant][cycle]["CD Training"]
            higher_AC = 'higher_errors_in_AC'
            if higher_AC not in errors_per_participant_per_cycle_per_phase[participant][cycle]:
                errors_per_participant_per_cycle_per_phase[participant][cycle][higher_AC] = []

            for errors_ab, errors_ac in zip(errors_AB, errors_AC):
                errors_per_participant_per_cycle_per_phase[participant][cycle][higher_AC].append(errors_ab > errors_ac )

    # now count how many times AB has more errors than AC per cycle
    cycle1_count = 0
    cycle2_count = 0
    cycle3_count = 0
    cycle4_count = 0
    cycle5_count = 0
    cycle6_count = 0
    cycle1_count_total = 0
    cycle2_count_total = 0
    cycle3_count_total = 0
    cycle4_count_total = 0
    cycle5_count_total = 0
    cycle6_count_total = 0

    for participant in errors_per_participant_per_cycle_per_phase:
        for cycle in errors_per_participant_per_cycle_per_phase[participant]:
            for higher_errors in errors_per_participant_per_cycle_per_phase[participant][cycle][higher_AC]:
                if cycle == 'Cycle 1':
                    cycle1_count_total += 1
                elif cycle == 'Cycle 2':
                    cycle2_count_total += 1
                elif cycle == 'Cycle 3':
                    cycle3_count_total += 1
                elif cycle == 'Cycle 4':
                    cycle4_count_total += 1
                elif cycle == 'Cycle 5':
                    cycle5_count_total += 1
                elif cycle == 'Cycle 6':
                    cycle6_count_total += 1
                if higher_errors:
                    if cycle == 'Cycle 1':
                        cycle1_count += 1
                    elif cycle == 'Cycle 2':
                        cycle2_count += 1
                    elif cycle == 'Cycle 3':
                        cycle3_count += 1
                    elif cycle == 'Cycle 4':
                        cycle4_count += 1
                    elif cycle == 'Cycle 5':
                        cycle5_count += 1
                    elif cycle == 'Cycle 6':
                        cycle6_count += 1

    print(f'Cycle 1: {cycle1_count} out of {cycle1_count_total}')
    print(f'Cycle 2: {cycle2_count} out of {cycle2_count_total}')
    print(f'Cycle 3: {cycle3_count} out of {cycle3_count_total}')
    print(f'Cycle 4: {cycle4_count} out of {cycle4_count_total}')
    print(f'Cycle 5: {cycle5_count} out of {cycle5_count_total}')
    print(f'Cycle 6: {cycle6_count} out of {cycle6_count_total}')

def multiple_probes_procedure(participants_list, group=False, append_to_filename='', save=True):
    print("You asked a group analysis for participants:", participants_list)
    container = []
    cd('..')
    for folder in participants_list:
        walk_and_execute(folder, collect_metadata, container, False, False)
    participants = list(set([data['identification'][0][1] for data in container]))
    print("Doing for participants:", participants)

    cd('analysis')
    cd('output')

    include_names = ['CD Probes 1', 'CD Probes 3']
    target_name = 'CD Probes'
    # group all hit_rates by name
    grouped_hit_rates = {}
    max_cycle = 0
    for data in container:
        identification = data['identification']
        categories = data['categories']
        # we need to group hit_rates per name then per cycle
        # cycle is the last element in identification tuple
        id_names, id_data = zip(*identification)
        max_cycle = max(max_cycle, int(id_data[3]))
        cycle = id_data[3]
        if cycle == '0':
            continue
        cycle = 'Cycle ' + cycle
        names, hit_rates, colors = zip(*categories)

        for name, hit_rate in zip(names, hit_rates):
            if name not in include_names:
                continue

            if hit_rate is None:
                continue

            if cycle not in grouped_hit_rates:
                grouped_hit_rates[cycle] = {target_name: [hit_rate]}
            grouped_hit_rates[cycle][target_name].append(hit_rate)

    # calculate average hit rate and sdt per cycle per name
    average_hit_rates = {}
    std_hit_rates = {}
    for cycle, hit_rates in grouped_hit_rates.items():
        for name, rates in hit_rates.items():
            if name not in average_hit_rates:
                average_hit_rates[name] = []
                std_hit_rates[name] = []
            average_hit_rates[name].append(np.mean(rates))
            std_hit_rates[name].append(np.std(rates))

    # we need a single plot with cycles at the x axis and hit_rates at the y axis, one line per name
    fig, ax = plt.subplots(figsize=(6, 3))
    default_axis_config(ax, False)
    ax.set_xlim(0, max_cycle + 1)
    ax.set_ylim(-10, 120)
    ax.set_xticks(range(1, max_cycle + 1))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Percent correct (Group main)')

    # plot one line per name for all cycles
    handles = []
    for name, hit_rates in average_hit_rates.items():
        x = np.array(range(1, max_cycle + 1))

        # use a circle for CD Probes 2, a triangle for AC Probes and a square for BC-CB Probes
        if name == target_name:
            label = 'Textual Beh'
            marker = 'o'
            markerfacecolor = 'gray'
        else:
            raise ValueError('Unknown name: {}'.format(name))

        # draw a dashed line at 0.5 and 1.0
        ax.axhline(y=50, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(y=100, color='black', linewidth=0.8, linestyle='--')

        hit_rates = [rate * 100 for rate in hit_rates]
        if group:
            upper_errors = std_hit_rates[name]
            lower_errors = [0] * len(upper_errors)
            upper_errors = [rate * 100 for rate in upper_errors]

            # Draw the error bars
            for i in range(len(x)):
                ax.vlines(x[i], hit_rates[i] - lower_errors[i], hit_rates[i] + upper_errors[i],
                        linewidth=1.0,
                        color='black')

            # Draw the caps
            for i in range(len(x)):
                ax.hlines(hit_rates[i] + upper_errors[i], x[i] - 0.1, x[i] + 0.1,
                        linewidth=1.0,
                        color='black')

        handle, = ax.plot(x, hit_rates,
                    markersize=8,
                    label=label,
                    color='black',
                    marker=marker,
                    linewidth=1.0,
                    markerfacecolor=markerfacecolor, markeredgecolor='black')
        handles.append(handle)

    # y ticks at 0, 0.5 and 1.0 only
    yticks = [0, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1f}'.format(y) for y in yticks])
    # show legend outside the plot
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    # tight layout
    plt.tight_layout()

    filename = 'line_per_cycle_CD'+append_to_filename+'.pdf'
    save_or_show(fig, save, filename)

    cd('..')

def multiple_probes_procedure_lines(participants_list, plot_type='line', group=False, append_to_filename=''):
    w1_filter = r'(nibo|fale)'
    w2_filter = r'(bofa|leni)'
    w3_filter = r'(lebo|fani)'
    w4_filter = r'(boni|lefa)'
    w5_filter = r'(fabo|nile)'
    w6_filter = r'(bole|nifa)'

    w7_filter = r'(bona|lefi)'
    w8_filter = r'(fabe|nilo)'
    w9_filter = r'(lani|febo)'
    w10_filter = r'(nole|bifa)'

    filters = [
        w1_filter, w2_filter, w3_filter, w4_filter, w5_filter, w6_filter,
        w7_filter, w8_filter, w9_filter, w10_filter]

    cd('output')
    filename = 'probes_CD.data.processed'
    CD_probes = load_probes_file(filename)
    CD_probes = CD_probes[CD_probes['Condition'] == 7]

    # need to treat some exceptions
    # must join Files "002.data.processed" and "006.data.processed" ONLY for for Participant "13-AND"
    CD_probes.loc[(CD_probes['Participant'] == '13-AND') & (CD_probes['File'] == '006.data.processed'), 'File'] = '002.data.processed'

    # must join Files "049.data.processed" and "050.data.processed" ONLY for for Participant "20-CAM"
    CD_probes.loc[(CD_probes['Participant'] == '20-CAM') & (CD_probes['File'] == '050.data.processed'), 'File'] = '049.data.processed'

    # must join Files "034.data.processed" and "037.data.processed" ONLY for for Participant "3-PVV "
    CD_probes.loc[(CD_probes['Participant'] == '3-PVV') & (CD_probes['File'] == '037.data.processed'), 'File'] = '034.data.processed'

    data_to_plot = {}
    for participant in participants_list:
        participant_probes = CD_probes[CD_probes['Participant'] == participant]
        unique_files = participant_probes['File'].unique()
        for file in unique_files:
            file_probes = participant_probes[participant_probes['File'] == file]
            # use the file index as a tag
            file_tag = unique_files.tolist().index(file) + 1
            if file_tag == 8:
                raise ValueError(f'index 8 not allowed: {participant}, {file}')
            if file_tag not in data_to_plot:
                data_to_plot[file_tag] = {}

            for filter in filters:
                word_probes = file_probes[file_probes['Name'].str.match(filter)]
                if word_probes.empty:
                    raise ValueError(f'No probes for {filter} in {file} for participant {participant}')

                if filter not in data_to_plot[file_tag]:
                    data_to_plot[file_tag][filter] = []

                # count trials and hits for each filter
                trials = word_probes['Result'].count()

                # count Hit in Result column
                hits = word_probes[word_probes['Result'] == 'Hit'].count()
                data_to_plot[file_tag][filter].append(hits / trials * 100)

    # plot the data, one ax per filter, x is the file index, y is the hit rate
    # share x axis
    fig, axs = plt.subplots(len(filters), 1, sharex=False, sharey=True, figsize=(3, 8))
    for i, filter in enumerate(filters):
        ax = axs[i]
        # remove outlines of axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # y_limit
        ax.set_ylim(-20, 120)
        y_label = filter.replace('(', '').replace(')', '').replace('|', '\n')
        ax.set_ylabel(y_label)

        # set y ticks at 0, 50, 100
        yticks = [0, 50, 100]
        ax.set_yticks(yticks)

        # remove x ticks for all but the last ax
        if i < len(filters) - 1:
            ax.set_xlim(0, len(data_to_plot) + 1)
            ax.set_xticks([])
        else:
            # x_limit
            ax.set_xlim(0, len(data_to_plot) + 1)
            ax.set_xticks(range(1, len(data_to_plot) + 1))

        # draw a dashed line at 0.0, 0.5 and 1.0
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.axhline(y=50, color='gray', linewidth=0.5, linestyle='--')
        ax.axhline(y=100, color='gray', linewidth=0.5, linestyle='--')

        # write participant name at the top/left of the plot
        if i == 0:
            ax.text(0, 125, participant)

        if plot_type == 'line':
            data_x = []
            data_y = []
            for file_tag in data_to_plot:
                # draw a vertical line at the file index
                if (file_tag == i+1) and (i < 6):
                    line_x = file_tag+0.5
                    ax.axvline(x=line_x, color='black', linewidth=1.0, linestyle='--')
                    ax.hlines(-20, line_x, line_x+1, color='black', linewidth=2.0, linestyle='--')

                y = np.mean(data_to_plot[file_tag][filter])
                x = file_tag
                # calculate std
                if group:
                    upper_error = y
                    lower_error = 0

                    # Draw the error bars
                    ax.vlines(x, y - lower_error, y + upper_error,
                            linewidth=1.0,
                            color='black')

                    # Draw the caps
                    ax.hlines(y + upper_error, x - 0.1, x + 0.1,
                            linewidth=1.0,
                            color='black')
                data_x.append(x)
                data_y.append(y)
            ax.plot(data_x, data_y, '-o', color='black', markersize=3.0)

        elif plot_type == 'box':
            print(data_to_plot[file_tag][filter])
            # data = [data_to_plot[file_tag][filter] for file_tag in data_to_plot]
            # ax.boxplot(data, showmeans=True, meanline=True)

    plt.tight_layout()

    # reduce the space between subplots vertically
    plt.subplots_adjust(hspace=0.1)

    save_or_show(fig, True, f'line_per_cycle_CD_probes{append_to_filename}.pdf')
    cd('..')


def sort_lines(participants_list):
    w1_filter = r'(nibo|fale)'
    w2_filter = r'(bofa|leni)'
    w3_filter = r'(lebo|fani)'
    w4_filter = r'(boni|lefa)'
    w5_filter = r'(fabo|nile)'
    w6_filter = r'(bole|nifa)'

    w7_filter = r'(bona|lefi)'
    w8_filter = r'(fabe|nilo)'
    w9_filter = r'(lani|febo)'
    w10_filter = r'(nole|bifa)'

    filters = [
        w1_filter, w2_filter, w3_filter, w4_filter, w5_filter, w6_filter,
        w7_filter, w8_filter, w9_filter, w10_filter]

    cd('output')
    filename = 'probes_CD.data.processed'
    CD_probes = load_probes_file(filename)
    CD_probes = CD_probes[CD_probes['Condition'] == 7]

    # need to treat some exceptions
    # must join Files "002.data.processed" and "006.data.processed" ONLY for for Participant "13-AND"
    CD_probes.loc[(CD_probes['Participant'] == '13-AND') & (CD_probes['File'] == '006.data.processed'), 'File'] = '002.data.processed'

    # must join Files "049.data.processed" and "050.data.processed" ONLY for for Participant "20-CAM"
    CD_probes.loc[(CD_probes['Participant'] == '20-CAM') & (CD_probes['File'] == '050.data.processed'), 'File'] = '049.data.processed'

    # must join Files "034.data.processed" and "037.data.processed" ONLY for for Participant "3-PVV "
    CD_probes.loc[(CD_probes['Participant'] == '3-PVV') & (CD_probes['File'] == '037.data.processed'), 'File'] = '034.data.processed'

    participants_data = {}
    for participant in participants_list:
        data_to_plot = {'hits':[], 'trials':[], 'probes':[]}
        participant_probes = CD_probes[CD_probes['Participant'] == participant]
        unique_files = participant_probes['File'].unique()
        for file in unique_files:
            file_probes = participant_probes[participant_probes['File'] == file]
            # use the file index as a tag
            file_tag = unique_files.tolist().index(file) + 1
            if file_tag == 8:
                raise ValueError(f'index 8 not allowed: {participant}, {file}')
            # if file_tag not in data_to_plot:
            #     data_to_plot[file_tag] = {}

            for filter in filters:
                word_probes = file_probes[file_probes['Name'].str.match(filter)]
                if word_probes.empty:
                    raise ValueError(f'No probes for {filter} in {file} for participant {participant}')

                # if filter not in data_to_plot[file_tag]:
                #     data_to_plot[file_tag][filter] = []

                # count trials and hits for each filter
                trials = word_probes['Result'].count()

                # count Hit in Result column
                hits = word_probes[word_probes['Result'] == 'Hit']['Result'].count()

                # data_to_plot[file_tag][filter].append((hits, trials))
                data_to_plot['hits'].append(hits)
                data_to_plot['trials'].append(trials)
                data_to_plot['probes'].append(word_probes)
        participants_data[participant] = {
            'hits': np.sum(data_to_plot['hits']),
            'trials': np.sum(data_to_plot['trials']),
            'proportion': np.sum(data_to_plot['hits']) / np.sum(data_to_plot['trials']),
            'probes': data_to_plot['probes']}

    # for participant in participants_data:
    #     data = []
    #     for file_tag in participants_data[participant]:
    #         for filter in participants_data[participant][file_tag]:
    #             data.append(participants_data[participant][file_tag][filter])
    #     participants_data[participant]['unfiltered'] = np.sum(np.array(data))

    return participants_data

if __name__ == '__main__':
    from constants import participants_pt as participants_folders

    # multiple_probes_procedure(
    #     participants_list=participant_folders,
    #     group=True,
    #     append_to_filename='_all')

    # for participant in participants_folders:
    #     multiple_probes_procedure_lines(
    #         participants_list=[participant],
    #         group=False,
    #         append_to_filename=f'_{participant}')

    participants_data = sort_lines(participants_list=participants_folders)
    sorted_participants = sorted(participants_data, key=lambda x: participants_data[x]['proportion'], reverse=True)
    for participant in sorted_participants:
        print(participant, participants_data[participant]['trials'], participants_data[participant]['proportion'])

    # sort participant_data by participants_data[participant]['unfiltered']
    # sorted_participants = sorted(participants_data, key=lambda x: participants_data[x]['unfiltered'], reverse=True)
    # for participant in sorted_participants:
    #     print(participant, participants_data[participant]['unfiltered'])
    
    # for participant in participants_data:
    #     print(participant, participants_data[participant]['unfiltered'])



    # line_plot_AB_AC_CD()
    # data = get_hits_trials_per_participant_per_cycle_per_phase(participants_list=participant_folders)
    # paired_chi_squared(data)
    # count_min_max(participants_list=participant_folders)
    # plot(participants_list=participant_folders)
    # count_errors()
    # plot_rank_tests()