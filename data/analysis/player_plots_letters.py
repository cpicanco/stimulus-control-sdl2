import os
import itertools
# from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np

from fileutils import data_dir, cd

from player_utils import calculate
from explorer import (
    cache_folder,
    foldername_2,
    load_from_csv
)

# teaching
nibo = 'nibo'
fale = 'fale'
boni = 'boni'
lefa = 'lefa'
nile = 'nile'
bole = 'bole'
fani = 'fani'
lebo = 'lebo'
bofa = 'bofa'
leni = 'leni'
nifa = 'nifa'
lebo = 'lebo'

# assessment
nale = 'nale'
lani = 'lani'
febo = 'febo'
nole = 'nole'
bifa = 'bifa'
nofa = 'nofa'
lofi = 'lofi'
fabe = 'fabe'
febi = 'febi'
lano = 'lano'
bena = 'bena'
nibe = 'nibe'
bofi = 'bofi'
leba = 'leba'
bona = 'bona'
lefi = 'lefi'
nilo = 'nilo'
falo = 'falo'

teaching = rf'({nibo}|{fale}|{boni}|{lefa}|{nile}|{bole}|{fani}{lebo}|{bofa}{leni}|{nifa}|{lebo})'
assessment = rf'({nale}|{lani}|{febo}|{nole}|{bifa}|{nofa}|{lofi}|{fabe}|{febi}|{lano}|{nibe}|{bofi}|{leba}|{bona}|{lefi}|{nilo})'

isr_left = rf'({nofa}|{nale}|{lani}|{febo}|{nole}|{bifa})'
isr_both = rf'({lofi}|{febi}|{lano}|{bena})'
isr_right = rf'({nibe}|{bofi}|{leba}|{bona}|{lefi}|{nilo}|{falo}|{fabe})'

list_right = [nibe, bofi, leba, bona, lefi, nilo, falo, fabe]
list_left = [nofa, nale, lani, febo, nole, bifa]
list_both = [lofi, febi, lano, bena]

def accumulated_frequency(data, hits, file_label, participant):
    """
    Returns the accumulated frequency of fixations over letter positions
    """
    data_dir()
    cd('analysis')
    cd('output')
    cd('fixations_over_letters')
    fig, axes = plt.subplots(1, 1)
    axes = [axes]

    fig.set_size_inches(5, 4)

    axes[0].set_ylabel('Frequência acumulada')

    maximum_y = 1100
    minimum_y = 0
    cummulative_lines = []
    for i, (trials, fixations, letter) in enumerate(data):
        # Create a list of datetimes from the list of strings

        # maximum_y = max(round(max(fixations)), maximum_y)

        # Create a stepped plot
        cummulative_line, = axes[0].step(trials, fixations,
                    linewidth=2,
                    where='post',
                    color=f'C{i}',
                    label=letter)

        cummulative_lines.append(cummulative_line)

    # Plot the additional line on the secondary y-axis
    ax2 = axes[0].twinx()
    # porcentage_line, = ax2.plot(hits['uid'], hits['target_calculation'],
    #                 linestyle='--',
    #                 linewidth=1,
    #                 marker='o',
    #                 color='black',
    #                 label='Percent Correct\n(4 trial groups)')

    porcentage_line = ax2.scatter(hits['uid'], hits['target_calculation'],
                    # linestyle='--',
                    # linewidth=1,
                    marker='o',
                    color='black',
                    label='Porcentagem de acerto\n(grupos de 4 tentativas)')


    # get the minimum and maximum
    min_val, max_val = minimum_y, maximum_y
    mid_val = (max_val + min_val) / 2
    # get absolute 10% of the range to  subtract from min and add to max
    abs_10 = (max_val - min_val) * 0.1
    yticks = [min_val, mid_val, max_val]
    for y in yticks:
        axes[0].axhline(y=y, color='black', linestyle='--', linewidth=0.5, zorder=0)
    axes[0].set_ylim([min_val - abs_10, max_val + abs_10])
    axes[0].set_yticks([min_val, mid_val//2, mid_val, mid_val+(mid_val//2), max_val])

    abs_10 = abs_10*100/max_val
    ax2.set_ylim(0-abs_10, 100+abs_10)

    axes.append(ax2)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Combine the legend entries from both axes
    handles = cummulative_lines
    labels = [line.get_label() for line in handles]

    # Add labels
    axes[0].set_xlabel('')
    ax2.set_ylabel('Porcentagem', color='black')

    # # Add x and y labels in the middle of figure
    fig.text(0.5, -0.02, 'Tentativas', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    fig.legend(handles, labels,
               title='Frequência acumulada de fixações sobre a letra',
               loc='upper left',
               ncol=4,
               columnspacing=1.5,
               handletextpad=0.5)

    fig.legend([porcentage_line], [porcentage_line.get_label()],
               loc='upper right',
               ncol=1,
               columnspacing=1.5,
               handletextpad=0.5)

    plt.subplots_adjust(top=0.88)
    # shade background for non-working periods
    # if holidays_ranges is not None:
    #     all_dates = sorted(set(all_dates))
    #     holidays_ranges = holidays_ranges[holidays_ranges['start'] >= min(all_dates)]
    #     holidays_ranges = holidays_ranges[holidays_ranges['end'] <= max(all_dates)]
    #     for i, row in holidays_ranges.iterrows():
    #         axes[0].axvspan(row['start'], row['end'], facecolor='gray', alpha=0.25)

    # Show the plot
    extension = 'svg'
    filename = f'{participant}_fixations_over_positions_{file_label}_accumulated.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()
    cd('..')
    cd('..')


def proportions(data, hits, file_label, participant):
    """
    Returns the proportion of fixations over letter positions
    """
    data_dir()
    cd('analysis')
    cd('output')
    cd('fixations_over_letters')
    fig, axes = plt.subplots(1, 1)
    axes = [axes]

    fig.set_size_inches(5, 4)

    axes[0].set_ylabel('Porcentagem')

    maximum_y = 100
    minimum_y = 0
    cummulative_lines = []
    for i, (trials, fixations, letter) in enumerate(data):
        # Create a list of datetimes from the list of strings

        # maximum_y = max(round(max(fixations)), maximum_y)

        # Create a stepped plot
        cummulative_line, = axes[0].plot(trials, fixations,
                    linewidth=2,
                    color=f'C{i}',
                    label=letter)

        cummulative_lines.append(cummulative_line)

    porcentage_line = axes[0].scatter(hits['uid'], hits['target_calculation'],
                    # linestyle='--',
                    # linewidth=1,
                    marker='o',
                    color='black',
                    label='Porcentagem de acertos\n(grupos de 4 tentativas)')

    # get the minimum and maximum
    min_val, max_val = minimum_y, maximum_y
    mid_val = (max_val + min_val) / 2
    # get absolute 10% of the range to  subtract from min and add to max
    abs_10 = (max_val - min_val) * 0.1
    yticks = [min_val, mid_val, max_val]
    for y in yticks:
        axes[0].axhline(y=y, color='black', linestyle='--', linewidth=0.5, zorder=0)
    axes[0].set_ylim([min_val - abs_10, max_val + abs_10])
    axes[0].set_yticks([min_val, mid_val//2, mid_val, mid_val+(mid_val//2), max_val])

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    handles = cummulative_lines+[porcentage_line]
    labels = [line.get_label() for line in handles]

    # Add labels
    axes[0].set_xlabel('')

    fig.text(0.5, -0.02, 'Tentativas', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    fig.legend(handles, labels,
               title='Fixações na letra',
               loc='upper left',
               ncol=4,
               columnspacing=1.5,
               handletextpad=0.5)

    plt.subplots_adjust(top=0.88)

    extension = 'svg'
    filename = f'{participant}_fixations_over_positions_{file_label}_proportion.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()
    cd('..')
    cd('..')


def single_index(data, hits, file_label, participant):
    """
    Returns the proportion of fixations over letter positions
    """
    data_dir()
    cd('analysis')
    cd('output')
    cd('fixations_over_letters')
    fig, axes = plt.subplots(2, 1)
    # axes = [axes]

    fig.set_size_inches(6, 5)

    axes[0].set_ylabel('Prevalence index')

    index_lines = []
    # for i, (trials, fixations, letter) in enumerate(data):
        # Create a list of datetimes from the list of strings

        # maximum_y = max(round(max(fixations)), maximum_y)

        # Create a stepped plot
    trials, index = data
    index_line, = axes[0].plot(trials, index,
                linewidth=2,
                color=f'C{0}',
                label='Prevalence of fixations\nover syllabels')

    index_lines.append(index_line)

    # Plot the additional line on the secondary y-axis
    ax2 = axes[0].twinx()
    # porcentage_line, = ax2.plot(hits['uid'], hits['target_calculation'],
    #                 linestyle='--',
    #                 linewidth=1,
    #                 marker='o',
    #                 color='black',
    #                 label='Percent Correct\n(4 trial groups)')

    # porcentage_line = ax2.scatter(hits['uid'], hits['target_calculation'],
    #                 # linestyle='--',
    #                 # linewidth=1,
    #                 marker='o',
    #                 color='black',
    #                 label='Percent correct\n(4 trial groups)')
    x = hits['uid']
    colors = ['white' if not val else 'black' for val in hits['result']]
    axes[1].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=50)



    # # get the minimum and maximum
    # min_val, max_val = -1, 1
    # mid_val = (max_val + min_val) / 2
    # # get absolute 10% of the range to  subtract from min and add to max
    # abs_10 = (max_val - min_val) * 0.1
    # yticks = [min_val, mid_val, max_val]
    # for y in yticks:
    #     axes[0].axhline(y=y, color='black', linestyle='--', linewidth=0.5, zorder=0)
    # axes[0].set_ylim([min_val - abs_10, max_val + abs_10])
    # axes[0].set_yticks([min_val, mid_val, max_val])
    # axes[0].set_yticklabels([f'Right ({min_val})', f'Both ({mid_val})', f'Left({max_val})'])

    # abs_10 = abs_10*100/max_val
    # ax2.set_ylim(0-abs_10, 100+abs_10)
    # ax2.set_yticks([0, 50, 100])
    # axes.append(ax2)


    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Combine the legend entries from both axes
    handles = index_lines
    labels = [line.get_label() for line in handles]

    # Add labels
    axes[0].set_xlabel('Trials')
    ax2.set_ylabel('Percent correct', color='black')

    # # Add x and y labels in the middle of figure
    fig.text(0.5, -0.02, 'Trials', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    fig.legend(handles, labels,
               loc='upper left',
               ncol=4,
               columnspacing=1.5,
               handletextpad=0.5)

    # fig.legend([porcentage_line], [porcentage_line.get_label()],
    #            loc='upper right',
    #            ncol=1,
    #            columnspacing=1.5,
    #            handletextpad=0.5)

    plt.subplots_adjust(top=0.88)
    # shade background for non-working periods
    # if holidays_ranges is not None:
    #     all_dates = sorted(set(all_dates))
    #     holidays_ranges = holidays_ranges[holidays_ranges['start'] >= min(all_dates)]
    #     holidays_ranges = holidays_ranges[holidays_ranges['end'] <= max(all_dates)]
    #     for i, row in holidays_ranges.iterrows():
    #         axes[0].axvspan(row['start'], row['end'], facecolor='gray', alpha=0.25)

    # Show the plot
    extension = 'png'
    filename = f'{participant}_fixations_over_positions_{file_label}.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()
    cd('..')
    cd('..')

def plot_accumulated_frequency():
    file_labels = ['01_cycle_teaching', '02_cycle_assessment', '03_multiple_probes_procedure']

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')
    df = pd.read_csv(filename)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)

    # example
    # sample1 : str = bena
    # sample1_letter1_fixations_count : int = fixation count over position 1 (b letter)
    # sample1_letter2_fixations_count : int = fixation count over position 2 (e letter)
    # sample1_letter3_fixations_count : int = fixation count over position 3 (n letter)
    # sample1_letter4_fixations_count : int = fixation count over position 4 (a letter)

    # normalize session column
    for participant, participant_data in df.groupby('participant'):
        # participant_data['file_num'] = participant_data['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        # unique_sorted = participant_data['file_num'].sort_values().unique()
        # mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        # participant_data['session'] = participant_data['file_num'].map(mapping) + 1
        # export(participant_data, '_teste.csv')
        filters = [
            (participant_data['condition'] == '2a') & (participant_data['relation'] == 'CD'),
            (participant_data['condition'] == '5'),
            (participant_data['condition'] == '7'),
        ]
        for file_label, filtering_logic in zip(file_labels, filters):
            cd_data = participant_data[filtering_logic]

            cd_data.loc[:, 'uid'] = range(1, len(cd_data) + 1)
            hits = calculate(cd_data, 'rolling_average_no_overlap', 'result')
            letters = ['1ª', '2ª', '3ª', '4ª']
            letters_count = [
                np.cumsum(cd_data['letter1_count'].values),
                np.cumsum(cd_data['letter2_count'].values),
                np.cumsum(cd_data['letter3_count'].values),
                np.cumsum(cd_data['letter4_count'].values)
            ]
            letters_trials = [
                [i for i in range(1, len(count) + 1)] for count in letters_count]

            data = zip(letters_trials, letters_count, letters)
            accumulated_frequency(data, hits, file_label, participant)

def plot_proportions():
    file_labels = ['01_cycle_teaching', '02_cycle_assessment', '03_multiple_probes_procedure']

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')
    df = pd.read_csv(filename)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)

    # example
    # sample1 : str = bena
    # sample1_letter1_fixations_count : int = fixation count over position 1 (b letter)
    # sample1_letter2_fixations_count : int = fixation count over position 2 (e letter)
    # sample1_letter3_fixations_count : int = fixation count over position 3 (n letter)
    # sample1_letter4_fixations_count : int = fixation count over position 4 (a letter)

    # normalize session column
    for participant, participant_data in df.groupby('participant'):
        # participant_data['file_num'] = participant_data['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        # unique_sorted = participant_data['file_num'].sort_values().unique()
        # mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        # participant_data['session'] = participant_data['file_num'].map(mapping) + 1
        # export(participant_data, '_teste.csv')
        filters = [
            (participant_data['condition'] == '2a') & (participant_data['relation'] == 'CD'),
            (participant_data['condition'] == '5'),
            (participant_data['condition'] == '7'),
        ]
        for file_label, filtering_logic in zip(file_labels, filters):
            cd_data = participant_data[filtering_logic]

            cd_data.loc[:, 'uid'] = range(1, len(cd_data) + 1)
            hits = calculate(cd_data, 'rolling_average_no_overlap', 'result')
            letters = ['1ª', '2ª', '3ª', '4ª']
            letters_count = [
                cd_data['letter1_count'].values,
                cd_data['letter2_count'].values,
                cd_data['letter3_count'].values,
                cd_data['letter4_count'].values
            ]

            letters_trials = [
                [i for i in range(1, len(count) + 1)] for count in letters_count]

            # Stack the letter fixation counts into a 2D array
            stacked_counts = np.vstack(letters_count)

            # Calculate the total fixation count for each iteration
            totals = stacked_counts.sum(axis=0)

            # Avoid division by zero by replacing totals of 0 with a small epsilon
            totals = np.where(totals == 0, 1e-8, totals)

            # Compute the proportions and convert to percentages
            percentages = (stacked_counts / totals) * 100

            # Split the proportions back into a list of arrays for each letter
            letters_proportion = [percentages[i] for i in range(4)]
            # Calculate moving average window=4 for each letter's proportions
            window_size = 10
            kernel = np.ones(window_size) / window_size  # Kernel [0.25, 0.25, 0.25, 0.25]
            letters_ma = []

            for prop in letters_proportion:
                n = len(prop)
                if n < window_size:
                    # Not enough trials: return array of NaNs
                    ma = np.full(n, np.nan)
                else:
                    # Compute moving averages for valid windows (positions 3 to end)
                    valid_ma = np.convolve(prop, kernel, mode='valid')
                    # Initialize array with NaNs
                    ma = np.full(n, np.nan)
                    # Fill valid moving averages starting at index 3
                    ma[window_size-1:] = valid_ma
                letters_ma.append(ma)

            data = zip(letters_trials, letters_ma, letters)
            proportions(data, hits, file_label, participant)


def plot_index():
    file_labels = ['01_cycle_teaching', '02_cycle_assessment', '03_multiple_probes_procedure']

    df = load_from_csv(foldername_2)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)

    filters_level_0 = [isr_left, isr_both, isr_right]
    # example
    # sample1 : str = bena
    # sample1_letter1_fixations_count : int = fixation count over position 1 (b letter)
    # sample1_letter2_fixations_count : int = fixation count over position 2 (e letter)
    # sample1_letter3_fixations_count : int = fixation count over position 3 (n letter)
    # sample1_letter4_fixations_count : int = fixation count over position 4 (a letter)

    # normalize session column
    for participant, participant_data in df.groupby('participant'):
        participant_data['file_num'] = participant_data['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        unique_sorted = participant_data['file_num'].sort_values().unique()
        mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        participant_data['session'] = participant_data['file_num'].map(mapping) + 1
        # export(participant_data, '_teste.csv')
        filters = [
            (participant_data['condition'] == '2a') & (participant_data['relation'] == 'CD'),
            (participant_data['condition'] == '5'),
            (participant_data['condition'] == '7'),
        ]
        participant_data = participant_data['']
        for file_label, filtering_logic in zip(file_labels, filters):
            cd_data = participant_data[filtering_logic]

            cd_data.loc[:, 'uid'] = range(1, len(cd_data) + 1)
            hits = calculate(cd_data, 'rolling_average_no_overlap', 'result')
            # letters = ['AOI1', 'AOI2', 'AOI3', 'AOI4']
            letters_count = [
                cd_data['sample1_letter1_fixations_count'].values + cd_data['sample1_letter2_fixations_count'].values,
                cd_data['sample1_letter3_fixations_count'].values + cd_data['sample1_letter4_fixations_count'].values,
            ]

            # Stack the letter fixation counts into a 2D array
            stacked_counts = np.vstack(letters_count)

            # Calculate the total fixation count for each iteration
            totals = stacked_counts.sum(axis=0)

            # Avoid division by zero by replacing totals of 0 with a small epsilon
            totals = np.where(totals == 0, 1e-8, totals)

            # Split the proportions back into a list of arrays for each letter
            index = (stacked_counts[0] - stacked_counts[1]) / totals

            data = ([i for i in range(1, len(index) + 1)], index)
            single_index(data, hits, file_label, participant)


def plot_index_isr_analysis():
    data_dir()
    cd(os.path.join('analysis', 'output', 'fixations_over_letters'))

    def single_index_isr(data, hits, file_label, participant):
        """
        Returns the proportion of fixations over letter positions
        """
        words = hits['word']
        responses = hits['response']
        fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        # axes = [axes]

        fig.set_size_inches(7, 4)

        axes[0].set_ylabel('Fixations over syllabels\n(prevalence)')

        lines = []
        # for i, (trials, fixations, letter) in enumerate(data):
            # Create a list of datetimes from the list of strings

            # maximum_y = max(round(max(fixations)), maximum_y)

            # Create a stepped plot
        trials, index = data[0]
        index_line, = axes[0].plot(trials, index,
                    linewidth=2,
                    color='k',
                    label='Fixations over syllabels (prevalence)')
        axes[0].grid(axis='x')
        # axes[0].set_xticks(trials)
        # axes[0].set_xticklabels(responses, rotation=45, ha='right')

        lines.append(index_line)

        # Plot the additional line on the secondary y-axis
        ax2 = axes[0].twinx()

        totals = data[1]
        ax2_maximum = max(totals)
        ax2_minimum = min(totals)
        fixations_line, = ax2.plot(trials, totals,
                    linewidth=1,
                    color='k',
                    linestyle='--',
                    label='Total fixations over word')
        lines.append(fixations_line)

        x = hits['uid']
        colors = ['white' if not val else 'black' for val in hits['result']]
        axes[1].grid(axis='x')
        axes[1].set_xticks(x)
        axes[1].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=50)
        axes[1].set_yticks([1])  # Single tick to center labels (optional)
        axes[1].set_yticklabels(['Oral Naming\nCorrect (black)\nIncorrect (white)'])
        # axes[1].set_ylim(0.5, 1.5)
        axes[1].set_xlabel('Words')
        axes[1].set_xticklabels(words, rotation=45, ha='right')

        # get the minimum and maximum
        min_val, max_val = -1, 1
        mid_val = (max_val + min_val) / 2
        # get absolute 10% of the range to  subtract from min and add to max
        abs_10 = (max_val - min_val) * 0.1
        yticks = [min_val, mid_val, max_val]
        for y in yticks:
            axes[0].axhline(y=y, color='black', linestyle='--', linewidth=0.5, zorder=0)
        axes[0].set_ylim([min_val - abs_10, max_val + abs_10])
        axes[0].set_yticks([min_val, mid_val, max_val])
        axes[0].set_yticklabels([f'Right ({min_val})', f'Both ({mid_val})', f'Left({max_val})'])

        abs_10 = abs_10*ax2_maximum/max_val
        ax2.set_ylim(ax2_minimum-abs_10, ax2_maximum+abs_10)
        ax2.set_yticks([ax2_minimum, ax2_maximum/2, ax2_maximum])

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)


        # Combine the legend entries from both axes
        handles = lines
        labels = [line.get_label() for line in handles]

        # Add labels
        ax2.set_ylabel('Total fixations over word', color='black')

        # # Add x and y labels in the middle of figure
        fig.text(0.5, -0.02, 'Trials', ha='center', va='center', fontsize=12)

        plt.tight_layout()
        fig.legend(handles, labels,
                loc='upper left',
                ncol=4,
                columnspacing=1.5,
                handletextpad=0.5)

        # fig.legend([porcentage_line], [porcentage_line.get_label()],
        #            loc='upper right',
        #            ncol=1,
        #            columnspacing=1.5,
        #            handletextpad=0.5)

        plt.subplots_adjust(top=0.88)
        # shade background for non-working periods
        # if holidays_ranges is not None:
        #     all_dates = sorted(set(all_dates))
        #     holidays_ranges = holidays_ranges[holidays_ranges['start'] >= min(all_dates)]
        #     holidays_ranges = holidays_ranges[holidays_ranges['end'] <= max(all_dates)]
        #     for i, row in holidays_ranges.iterrows():
        #         axes[0].axvspan(row['start'], row['end'], facecolor='gray', alpha=0.25)

        # Show the plot
        extension = 'png'
        filename = f'isr_{participant}_fixations_over_positions_{file_label}.{extension}'
        print(filename)
        plt.savefig(filename, format=extension, dpi=300, transparent=False)
        plt.close()

    def draw(filtered_data, participant, file_label, file_label2):
        filtered_data.loc[:, 'uid'] = range(1, len(filtered_data) + 1)
        # letters = ['AOI1', 'AOI2', 'AOI3', 'AOI4']
        letters_count = [
            filtered_data['letter1_count'].values + filtered_data['letter2_count'].values,
            filtered_data['letter3_count'].values + filtered_data['letter4_count'].values,
        ]


        # Stack the letter fixation counts into a 2D array
        stacked_counts = np.vstack(letters_count)
        # Calculate the total fixation count for each iteration
        totals = stacked_counts.sum(axis=0)
        # Avoid division by zero
        valid = totals != 0
        stacked_counts = stacked_counts[:, valid]
        totals = totals[valid]
        # Split the proportions back into a list of arrays for each letter
        index = (stacked_counts[0] - stacked_counts[1]) / totals

        data = ([i for i in range(1, len(index) + 1)], index)

        single_index_isr(
            [data, totals],
            filtered_data,
            '_'.join([file_label, file_label2]),
            participant)


    file_labels = ['02_cycle_assessment', '03_multiple_probes_procedure']

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')
    df = pd.read_csv(filename)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)

    file_labels_2 = ['left', 'both', 'right']
    filters_level_1 = [isr_left, isr_both, isr_right]
    lists_of_words = [list_left, list_both, list_right]

    # normalize session column
    for participant, participant_data in df.groupby('participant'):

        filters_level_0 = [
            (participant_data['condition'] == '5'),
            (participant_data['condition'] == '7'),
        ]

        for file_label, filter_0 in zip(file_labels, filters_level_0):
            cd_data = participant_data[filter_0]
            for file_label2, filter_1, list_of_words in zip(file_labels_2, filters_level_1, lists_of_words):
                filtered_data = cd_data[cd_data['word'].str.match(filter_1)].copy(deep=True)
                if filtered_data.empty:
                    if file_label2 == 'both' and file_label == '03_multiple_probes_procedure':
                        continue
                    raise ValueError(f"No data found for {participant}, {file_label}, {file_label2}")

                draw(filtered_data, participant, file_label, file_label2)

                for word in list_of_words:
                    pattern = rf'{word}'
                    filtered_data = cd_data[cd_data['word'].str.match(pattern)].copy(deep=True)
                    if filtered_data.empty:
                        if word == lani and file_label == '02_cycle_assessment':
                            continue

                        if word == febo and file_label == '02_cycle_assessment':
                            continue

                        if word == nole and file_label == '02_cycle_assessment':
                            continue

                        if word == bifa and file_label == '02_cycle_assessment':
                            continue

                        if word == nofa and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == lofi and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == febi and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == lano and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == bena and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == falo and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == nale and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == bofi and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == nibe and file_label == '03_multiple_probes_procedure':
                            continue

                        if word == leba and file_label == '03_multiple_probes_procedure':
                            continue

                        raise ValueError(f"No data found for {participant}, {file_label}, {file_label2}, {word}")

                    label_2 = file_label2 + '_' + word
                    draw(filtered_data, participant, file_label, label_2)

def calculate_index(row,
                    col1='sample1_letter1_fixations_count',
                    col2='sample1_letter2_fixations_count',
                    col3='sample1_letter3_fixations_count',
                    col4='sample1_letter4_fixations_count',
                    use_delta=True):
    left = row[col1] + row[col2]
    right = row[col3] + row[col4]
    total = left + right
    if total != 0:
        if use_delta:
            return (left - right) / total
        else:
            return left / total
    else:
        return np.nan  # Handle rows with total = 0 (e.g., NaN, 0, or other)

def compare_correct_responses_in_isr_categories(df, target_participant=''):
    if target_participant != '':
        df = df[df['participant'] == target_participant].copy(deep=True)

    df = df[(df['condition'] == '5') | (df['condition'] == '7')].copy(deep=True)
    # Extract file numbers
    df.loc[:, 'file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)

    # Create session numbers within each participant group
    df.loc[:, 'session'] = (df.groupby('participant')['file_num']
                    .rank(method='dense')
                    .astype(int))

    # Identify the last session for each participant
    max_sessions = df.groupby('participant')['session'].transform('max')

    # Update 'cycle' to 7 where the session is the last one for the participant
    df.loc[df['session'] == max_sessions, 'cycle'] = 7

    # df = df[df['result'] == 1]
    print(len(df))

    filters = [isr_left, isr_both, isr_right, teaching]

    condition_map = {0: '5', 1: '7'}
    condition_data = []
    contition_hits = []
    for condition in condition_map.keys():
        target_df = df[(df['condition'] == condition_map[condition])]
        cycle_data = []
        cycle_hits = []
        for _, dataframe in target_df.groupby('cycle'):
            filtered = [dataframe[dataframe['word'].str.match(filter_)] for filter_ in filters]

            total_sum = 0
            total_len = 0
            for df_ in filtered:
                total_sum += np.sum(df_['result'])
                total_len += len(df_['result'])
            cycle_hits.append((total_sum/total_len)*100)

            processed_data = []
            for df_ in filtered:
                letters_count = [
                    df_['sample1_letter1_fixations_count'].values + df_['sample1_letter2_fixations_count'].values,
                    df_['sample1_letter3_fixations_count'].values + df_['sample1_letter4_fixations_count'].values,
                ]

                stacked_counts = np.vstack(letters_count)
                totals = stacked_counts.sum(axis=0)
                valid = totals != 0
                stacked_counts = stacked_counts[:, valid]
                totals = totals[valid]
                index = (stacked_counts[0] - stacked_counts[1]) / totals

                # data = ([i for i in range(1, len(index) + 1)], index, totals)
                # print(len(index))
                processed_data.append(index.T)
            cycle_data.append(processed_data)
        condition_data.append((cycle_data, len(cycle_data)))
        contition_hits.append(cycle_hits)

    num_filters = len(filters)

    # draw a box plot of indexes for each isr category
    if participant != '':
        fig, axes = plt.subplots(2, 2,
                                 figsize=(10, 5),
                                 sharex=False,
                                 sharey=False,
                                gridspec_kw={
                                'height_ratios': [3, 1],  # Top row (3), Bottom row (1)
                                'width_ratios': [2.5, 3],   # Left column (2), Right column (3)
                            })
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=False, sharey=True)
    axes = axes.flatten()
    for condition, (ax, condition_data) in enumerate(zip(axes, condition_data)):
        cycle_data, num_cycles = condition_data
        if (condition == 1) and (num_cycles == 6):
            # add empty nan to fill the gap in the first cycle
            cycle_data.insert(0, [[] for _ in range(num_filters)])
            num_cycles += 1

        for i, data in enumerate(cycle_data):
            # Calculate positions for this cycle's group
            start_pos = i * (num_filters + 1)  # +1 adds spacing between cycles
            positions = [start_pos + j for j in range(num_filters)]
            # Create boxplots with the calculated positions
            bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

            # Optional: Assign colors to each filter for better distinction
            colors = plt.cm.Pastel1.colors[:num_filters]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            # Set x-axesis ticks and labels
            xticks_positions = [i*(num_filters + 1) + (num_filters - 1)/2 for i in range(num_cycles)]
            ax.set_xticks(xticks_positions)
            ax.set_xticklabels([f'{i+1}' for i in range(num_cycles)])


    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    if participant != '':
        axes[2].plot(contition_hits[0], linewidth=2, color='k')
        axes[2].set_xticks([i for i in range(6)])
        axes[2].set_xticklabels([f'{i+1}' for i in range(6)])

        axes[3].plot(contition_hits[1], linewidth=2, color='k')
        axes[3].set_xticks([i for i in range(7)])
        axes[3].set_xticklabels([f'{i+1}' for i in range(7)])

        for ax in axes[2:]:
            ax.axhline(y=[100], color='black', linestyle='--', linewidth=0.5, zorder=0)
            ax.axhline(y=[50], color='black', linestyle='--', linewidth=0.5, zorder=0)
            ax.axhline(y=[0], color='black', linestyle='--', linewidth=0.5, zorder=0)


    for ax in axes[:2]:
        ax.axhline(y=[1], color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax.axhline(y=[.5], color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax.axhline(y=[0], color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax.axhline(y=[-.5], color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax.axhline(y=[-1], color='black', linestyle='--', linewidth=0.5, zorder=0)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([f'Right ({-1})', f'Both ({0})', f'Left({1})'])


    # Add legend for filters
    legend_labels = ['Left', 'Both', 'Right', 'Teaching']
    colors = plt.cm.Pastel1.colors[:num_filters]
    fig.legend(
        handles=[plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(num_filters)],
        labels=legend_labels, title='Intrasyllabic recombination position',
        # bbox_to_anchor=(1, 1),
        ncol=4,
        loc='upper center'
        )

    if participant != '':
        axes[0].set_ylabel('Left/Right proportion')
        axes[2].set_ylabel('Percent correct')
        axes[2].set_xlabel('Cycles')
        axes[3].set_xlabel('Probes')
    else:
        axes[0].set_xlabel('Cycles')
        axes[1].set_xlabel('Probes')
        # add y label in the center of both axes
        fig.text(0.05, 0.5, 'Left/Right proportion', ha='center', va='center', rotation=90, fontsize=10)


    # write participant name at top left
    fig.text(0.05, 0.95, f'{target_participant.replace('\\', '')}', ha='left', va='center', fontsize=10)

    plt.tight_layout(rect=[0.05, 0.05, 1., 0.9])  # Right margin at 85% of figure width
    # save to file
    # Show the plot
    extension = 'png'
    filename = f'{target_participant.replace('\\', '')}_irp_proportion.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close(fig)

def compare_correct_incorrect_density():
    import seaborn as sns
    import scipy.stats as stats

    from explorer import (
        foldername_1,
        foldername_3)
    df1 = load_from_csv(foldername_1)
    df2 = load_from_csv(foldername_2)
    df3 = load_from_csv(foldername_3)
    df = pd.concat([df1, df2, df3])
    df['index'] = df.apply(calculate_index, axis=1)
    df = df[df['cycle'] == 2]


    hits = {'participant': [], 'hits': [], 'index': []}
    for participant in df['participant'].unique():
        cd_probes = df[(df['participant'] == participant)].copy(deep=True)
        cd_probes = cd_probes[(cd_probes['condition'] == '5') | (cd_probes['condition'] == '7')]

        total_sum = np.sum(cd_probes['result'])
        total_len = len(cd_probes['result'])
        hits['participant'].append(participant)
        hits['hits'].append((total_sum / total_len)*100)
        hits['index'].append(np.mean(cd_probes['index']))

    hits = pd.DataFrame(hits)
    hits['participant'] = hits['participant'].apply(lambda x: x.replace('\\', ''))
    # hits = hits.sort_values(by='hits', ascending=False)
    # hits['hits'].plot.hist(bins=18)

    # plt.xticks(rotation=45, ha='right')
    # # plt.xlabel('Participant')
    # # plt.ylabel('Percent correct')
    # # plt.ylim(0, 100)
    # plt.legend().remove()
    # plt.tight_layout(rect=[0.05, 0.05, 1., 0.9])
    # plt.show()

    # Assuming 'df' is your DataFrame and it has a 'result' column
    sns.histplot(data=df, x='index', bins=20, kde=True, hue='result', multiple='layer')  # 'multiple' controls how histograms are displayed

    # write text with t-students t-test with confidence interval
    t, p = stats.ttest_ind(df[df['result'] == 0]['index'], df[df['result'] == 1]['index'])

    plt.text(0.5, 0.9, f'T-test: t={t:.2f}, p={p:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

    # change lagend title
    plt.legend(title='Result', labels=['Incorrect', 'Correct'])

    plt.xlim(1, -1)
    plt.xlabel('Left/Right proportion')
    plt.show()

    # for participant in df['participant'].unique():
    #     compare_correct_responses_in_isr_categories(df, participant)
    # plot_index_isr_analysis()

def compare_correct_incorrect_density_in_isr_categories():
    import seaborn as sns
    import scipy.stats as stats

    df = load_from_csv(foldername_2)
    df.loc[:, 'score'] = df.apply(calculate_index, axis=1)
    df.loc[:, 'file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
    df.loc[:, 'session'] = (df.groupby('participant')['file_num']
                    .rank(method='dense')
                    .astype(int))
    max_sessions = df.groupby('participant')['session'].transform('max')
    df.loc[df['session'] == max_sessions, 'cycle'] = 7

    conditions = [
        df['word'].str.match(isr_left),
        df['word'].str.match(isr_right),
        df['word'].str.match(isr_both),
        df['word'].str.match(teaching)
    ]

    groups = [
        'left',
        'right',
        'both',
        'none'
    ]

    df['group'] = np.select(conditions, groups, default='other')

    for participant in df['participant'].unique():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, result in zip(axes, [0, 1]):
            cd_probes = df[
                        (df['participant'] == participant) &
                        (df['condition'] == '7') &
                        (df['result'] == result)].copy()

            # Split into left/right
            left = cd_probes[cd_probes['group'] == 'left']
            right = cd_probes[cd_probes['group'] == 'right']

            sns.histplot(data=left, x='score', bins=20, kde=True, color='red', alpha=0.5, label='Left', ax=ax)
            sns.histplot(data=right, x='score', bins=20, kde=True, color='blue', alpha=0.5, label='Right', ax=ax)

            ax.set_xlim(1.1, -1.1)
            ax.set_xlabel('Looked at left | Proportion | Looked at right')
            if result == 0:
                text = 'Incorrect naming'
            else:
                text = 'Correct naming'

            # write 'Correct naming' on top left
            ax.text(x=0.05, y=0.95, s=text,
                    ha='left', va='top', transform=ax.transAxes)

            # Perform Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(
                left['score'], right['score'], )
            mean_left = left['score'].mean()
            mean_right = right['score'].mean()
            ax.axvline(x=mean_left, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=mean_right, color='blue', linestyle='--', alpha=0.5)

            # Add statistical annotation
            text = (f'Mann-Whitney U = {stat:.3f}, p = {p_value:.3f}\n'
                    f'Left Mean (N={len(left)})= {mean_left:.2f}     Right Mean (N={len(right)})= {mean_right:.2f}')
            ax.text(x=1.0, y=1.13, s=text,
                    ha='right', va='top', transform=ax.transAxes)

        ax.legend(
            title='New syllable at',
            labels=['left', 'right'],
            loc='upper right',
            ncol=1)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        extension = 'png'
        participant = participant.replace('\\', '')
        filename = f'isr_{participant}_fixations_over_positions_density.{extension}'
        print(filename)
        plt.savefig(filename, format=extension, dpi=300, transparent=False)
        plt.close()

def compare_correct_incorrect_density_in_isr_categories_group():
    import seaborn as sns
    import scipy.stats as stats

    df = load_from_csv(foldername_2)
    df.loc[:, 'score'] = df.apply(calculate_index, axis=1)
    df.loc[:, 'file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
    df.loc[:, 'session'] = (df.groupby('participant')['file_num']
                    .rank(method='dense')
                    .astype(int))
    max_sessions = df.groupby('participant')['session'].transform('max')
    df.loc[df['session'] == max_sessions, 'cycle'] = 7

    conditions = [
        df['word'].str.match(isr_left),
        df['word'].str.match(isr_right),
        df['word'].str.match(isr_both),
        df['word'].str.match(teaching)
    ]

    groups = [
        'left',
        'right',
        'both',
        'none'
    ]

    df['group'] = np.select(conditions, groups, default='other')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, result in zip(axes, [0, 1]):
        cd_probes = df[
                    (df['condition'] == '7') &
                    (df['result'] == result)].copy()

        # Split into left/right
        left = cd_probes[cd_probes['group'] == 'left']
        right = cd_probes[cd_probes['group'] == 'right']

        sns.histplot(data=left, x='score', bins=20, kde=True, color='red', alpha=0.5, label='Left', ax=ax)
        sns.histplot(data=right, x='score', bins=20, kde=True, color='blue', alpha=0.5, label='Right', ax=ax)

        ax.set_xlim(1.1, -1.1)
        ax.set_xlabel('Looked at left | Proportion | Looked at right')
        if result == 0:
            text = 'Incorrect naming'
        else:
            text = 'Correct naming'

        # write 'Correct naming' on top left
        ax.text(x=0.05, y=0.95, s=text,
                ha='left', va='top', transform=ax.transAxes)

        # Perform Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(
            left['score'], right['score'], )
        mean_left = left['score'].mean()
        mean_right = right['score'].mean()
        ax.axvline(x=mean_left, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=mean_right, color='blue', linestyle='--', alpha=0.5)

        # Add statistical annotation
        text = (f'Mann-Whitney U = {stat:.3f}, p = {p_value:.3f}\n'
                f'Left Mean (N={len(left)})= {mean_left:.2f}     Right Mean (N={len(right)})= {mean_right:.2f}')
        ax.text(x=1.0, y=1.13, s=text,
                ha='right', va='top', transform=ax.transAxes)

    ax.legend(
        title='New syllable at',
        labels=['left', 'right'],
        loc='upper right',
        ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    extension = 'png'
    filename = f'isr_fixations_over_positions_density.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()

def plot_densities(df, target_column, split_correct_incorrect=False):
    def draw(ax, df, target_column):
        # Split into left/right
        left = df[df['group'] == 'left']
        right = df[df['group'] == 'right']

        sns.histplot(data=left, x=target_column, bins=20, kde=True, color='red', alpha=0.5, label='Left', ax=ax)
        sns.histplot(data=right, x=target_column, bins=20, kde=True, color='blue', alpha=0.5, label='Right', ax=ax)

        ax.set_xlim(1.1, -1.1)
        ax.set_xlabel('Looked at left | Proportion | Looked at right')
        if result == 0:
            text = 'Incorrect naming'
        else:
            text = 'Correct naming'

        # write 'Correct naming' on top left
        ax.text(x=0.05, y=0.95, s=text,
                ha='left', va='top', transform=ax.transAxes)

        # Perform Mann-Whitney U test
        # stat, p_value = stats.mannwhitneyu(
        #     left['score'], right['score'], )
        mean_left = left[target_column].mean()
        mean_right = right[target_column].mean()
        ax.axvline(x=mean_left, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=mean_right, color='blue', linestyle='--', alpha=0.5)

        # Add statistical annotation
        # text = (f'Mann-Whitney U = {stat:.3f}, p = {p_value:.3f}\n'
        #         f'Left Mean (N={len(left)})= {mean_left:.2f}     Right Mean (N={len(right)})= {mean_right:.2f}')
        text = (f'Left Mean (N={len(left)})= {mean_left:.2f}     Right Mean (N={len(right)})= {mean_right:.2f}')
        ax.text(x=1.0, y=1.13, s=text,
                ha='right', va='top', transform=ax.transAxes)

    if split_correct_incorrect:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, result in zip(axes, [0, 1]):
            draw(ax, df[df['result'] == result], target_column)
        ax.legend(
            title='New syllable at',
            labels=['left', 'right'],
            loc='upper right',
            ncol=1)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        draw(ax, df, target_column)
        ax.legend(
            title='New syllable at',
            labels=['left', 'right'],
            loc='upper right',
            ncol=1)

    postfix_map = {True:'correct_incorrect', False:'all'}

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    extension = 'png'
    filename = f'isr_{target_column}_density_{postfix_map[split_correct_incorrect]}.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()


def plot_index_isr_analysis_v2():
    data_dir()
    cd(os.path.join('analysis', 'output', 'fixations_over_letters_v2'))

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')
    df = pd.read_csv(filename)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)
    df.loc[:, 'score'] = df.apply(calculate_index, axis=1, use_delta=False)
    df.loc[:, 'score_delta'] = df.apply(calculate_index, axis=1, use_delta=True)
    df.loc[:, 'file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
    df.loc[:, 'session'] = (df.groupby('participant')['file_num']
                    .rank(method='dense')
                    .astype(int))
    max_sessions = df.groupby('participant')['session'].transform('max')
    df.loc[df['session'] == max_sessions, 'cycle'] = 7

    conditions = [
        df['word'].str.match(isr_left),
        df['word'].str.match(isr_right),
        df['word'].str.match(isr_both),
        df['word'].str.match(teaching)
    ]

    groups = [
        'left',
        'right',
        'both',
        'none'
    ]

    df['group'] = np.select(conditions, groups, default='other')


    # df = df[
    #     ((df['condition'] == '7') | (df['condition'] == '5')) &
    #     ((df['group'] == 'left') | (df['group'] == 'right'))].copy(deep=True)

    # hits_overall_df = df.pivot_table(
    #     index='participant',
    #     columns='group',
    #     values='result',
    #     aggfunc='sum'
    # )


    df = df[
        (df['condition'] == '7')].copy(deep=True)

    df['switchings_per_seconds'] = df['sample1_letters_switchings'] / df['sample_duration']


    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.histplot(data=df[(df['group'] == 'left') & (df['result'] == 1)], x='switchings_per_seconds', bins=20, kde=True, color='red', alpha=0.5, label='Left', ax=ax)
    # sns.histplot(data=df[(df['group'] == 'right') & (df['result'] == 1)], x='switchings_per_seconds', bins=20, kde=True, color='blue', alpha=0.5, label='Right', ax=ax)
    # sns.histplot(data=df[(df['group'] == 'none') & (df['result'] == 1)], x='switchings_per_seconds', bins=20, kde=True, color='green', alpha=0.5, label='Right', ax=ax)

    # # ax.set(xlim=(1, -1))
    # plt.xlabel('switchings_per_seconds')
    # plt.ylabel('Count')
    # plt.tight_layout()
    # plt.show()





    hits_df = df.pivot_table(
        index='participant',
        columns='group',
        values='result',
        aggfunc='sum'
    )

    mean_df = df.pivot_table(
        index='participant',
        columns='group',
        values='score',
        aggfunc='mean'
    )

    switchings_df = df.pivot_table(
        index='participant',
        columns='group',
        values='sample1_letters_switchings',
        aggfunc='mean'
    )
    switchings_df = switchings_df.rename(columns={''
        'left': 'left_switchings',
        'right': 'right_switchings',
        'none': 'none_switchings'})

    switchings_df['new_switchings'] = switchings_df['left_switchings'] + switchings_df['right_switchings']
    switchings_df['total_switchings'] = (switchings_df['left_switchings'] + switchings_df['right_switchings'] + switchings_df['none_switchings'])
    switchings_df['switching_ratio'] = switchings_df['new_switchings'] / switchings_df['total_switchings']



    # mean_delta_df = df.pivot_table(
    #     index='participant',
    #     columns='group',
    #     values='score_delta',
    #     aggfunc='mean'
    # )
    # mean_delta_df = mean_delta_df.rename(columns={''
    #     'left': 'left_delta',
    #     'right': 'right_delta',
    #     'none': 'none_delta'})

    # count_df = df.pivot_table(
    #     index='participant',
    #     columns='group',
    #     values='score',
    #     aggfunc='count'
    # )

    # hits_df = hits_df.rename(columns={''
    #     'left': 'left_hits',
    #     'right': 'right_hits',
    #     'none': 'none_hits'})
    # hits_df['total_hits'] = (hits_df['left_hits'] + hits_df['right_hits'] + hits_df['none_hits'])

    # count_df = count_df.rename(columns={
    #     'left': 'left_count',
    #     'right': 'right_count',
    #     'both': 'both_count',
    #     'none': 'none_count'})
    # count_df['total_count'] = (count_df['left_count'] + count_df['right_count'] + count_df['none_count'])


    # delta_df = pd.concat([mean_df, count_df, hits_df, mean_delta_df, switchings_df], axis=1)
    # delta_df['difference'] = delta_df['left'] - delta_df['right']
    # delta_df = delta_df.reset_index()
    # delta_df = delta_df.sort_values(by='difference', ascending=False)

    # # print mean of delta_df['difference']
    # print(f'Index Mean: {delta_df['difference'].mean()}')
    # print(f'Index Min.: {delta_df['difference'].min()}')
    # print(f'Index Max.: {delta_df['difference'].max()}')

    # delta_df['hit_proportion'] = delta_df['total_hits'] / delta_df['total_count']

    # print(f'Hit Proportion Mean: {delta_df["hit_proportion"].mean()}')
    # print(f'Hit Proportion Min.: {delta_df["hit_proportion"].min()}')
    # print(f'Hit Proportion Max.: {delta_df["hit_proportion"].max()}')

    # print(f'Teaching Mean: {delta_df["none_delta"].mean()}')
    # print(f'Teaching Min.: {delta_df["none_delta"].min()}')
    # print(f'Teaching Max.: {delta_df["none_delta"].max()}')

    # # list delta_df["hit_proportion"] per participant
    # for participant in delta_df["participant"].unique():
    #     print(f'Participant {participant} hit_proportion: {delta_df[delta_df["participant"] == participant]["hit_proportion"].mean()}')


    # y = delta_df['hit_proportion'].to_list()
    # x = delta_df['difference'].to_list()

    # from correlation import plot_correlation
    # plot_correlation(x, y,
    #                  'New syllable tracking during assessment',
    #                  'Hit proportion',
    #                  'left_right_delta',
    #                  save=True)


    # x = delta_df['none_delta'].to_list()

    # plot_correlation(x, y,
    #                  'Teached syllable tracking during assessment',
    #                  'Hit proportion',
    #                  'teaching_delta',
    #                  save=True)



    # # exclude participants with left_count or right_count < 10
    # delta_df = delta_df[(delta_df['left_count'] >= 10) & (delta_df['right_count'] >= 10)]

    # # Get the sorted participant order from delta_df
    # participant_order = delta_df['participant'].tolist()
    # difference_order = delta_df['difference'].tolist()

    # # Convert 'participant' to a categorical type with the custom order
    # df_sorted = df.copy()
    # df_sorted['participant'] = pd.Categorical(
    #     df_sorted['participant'],
    #     categories=participant_order,  # Enforce the desired order
    #     ordered=True
    # )

    # # Sort the DataFrame by the categorical 'participant' order
    # df_sorted = df_sorted.sort_values('participant')

    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['session', 'uid_in_session'])

    #         trials = [i+1 for i in range(len(group_data))]
    #         score = group_data['score'].tolist()

    #         # apply a moving average to smooth the data
    #         window_size = 4
    #         moving_average = np.convolve(score, np.ones(window_size)/window_size, mode='valid')
    #         # Calculate how many NaNs to pad (for odd window_size)
    #         n_pad = window_size - 1

    #         # Pad with NaNs at the beginning
    #         score = np.concatenate([
    #             np.full(n_pad, np.nan),
    #             moving_average])

    #         if group == 'left':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         color='k',
    #                         label='New syllable on the left')

    #         elif group == 'right':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='k',
    #                         label='New syllable on the right')
    #         lines.append(line)

    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     axes[n].set_ylim(0, 1)
    #     axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(4, len(trials)+1, 4))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1
    # # exclude empty axes from fig
    # # axes = [ax for ax in axes if not ax.lines]
    # # for ax in axes:
    # #     fig.delaxes(ax)

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Proportion of fixations on the left syllable', ha='center', va='center', fontsize=12, rotation=90)

    # handles = [lines[1], lines[0]]
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=4,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()

    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['session', 'uid_in_session'])
    #         trials = [i+1 for i in range(len(group_data))]
    #         score = group_data['result'].tolist()
    #         score = list(itertools.accumulate(score))

    #         if group == 'left':
    #             line, = axes[n].step(trials, score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         where='post',
    #                         color='k',
    #                         label='New syllable on the left')

    #         elif group == 'right':
    #             line, = axes[n].step(trials, score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         where='post',
    #                         color='gray',
    #                         label='New syllable on the right')
    #         lines.append(line)

    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(4, len(trials)+1, 4))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1
    # # exclude empty axes from fig
    # # axes = [ax for ax in axes if not ax.lines]
    # # for ax in axes:
    # #     fig.delaxes(ax)

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Proportion of fixations on the left syllable', ha='center', va='center', fontsize=12, rotation=90)

    # handles = [lines[1], lines[0]]
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=4,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_correct.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()


    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['session', 'uid_in_session'])
    #         trials = [i+1 for i in range(len(group_data))]
    #         score = group_data['sample1_letters_switchings'].tolist()
    #         # score = list(itertools.accumulate(score))

    #         if group == 'left':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         color='k',
    #                         label='New syllable on the left')

    #         elif group == 'right':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='gray',
    #                         label='New syllable on the right')
    #         lines.append(line)

    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(4, len(trials)+1, 4))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1
    # # exclude empty axes from fig
    # # axes = [ax for ax in axes if not ax.lines]
    # # for ax in axes:
    # #     fig.delaxes(ax)

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Switchings over letters', ha='center', va='center', fontsize=12, rotation=90)

    # handles = [lines[1], lines[0]]
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=4,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_switchings.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()


    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['session', 'uid_in_session'])
    #         trials = [i+1 for i in range(len(group_data))]
    #         score = group_data['latency'].tolist()
    #         # score = list(itertools.accumulate(score))

    #         if group == 'left':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         color='k',
    #                         label='New syllable on the left')

    #         elif group == 'right':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='gray',
    #                         label='New syllable on the right')
    #         lines.append(line)

    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(4, len(trials)+1, 4))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1
    # # exclude empty axes from fig
    # # axes = [ax for ax in axes if not ax.lines]
    # # for ax in axes:
    # #     fig.delaxes(ax)

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Oral naming latency', ha='center', va='center', fontsize=12, rotation=90)

    # handles = [lines[1], lines[0]]
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=4,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_latency.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()





    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['session', 'uid_in_session'])
    #         trials = [i+1 for i in range(len(group_data))]
    #         # score = group_data['result'].tolist()

    #         tdf = group_data.copy(deep=True)
    #         tdf['uid'] = range(1, len(tdf) + 1)
    #         tdf['group'] = (tdf['uid'] - 1)  // 4
    #         tdf['target_calculation'] = tdf['result'].groupby(tdf['group']).transform('mean') * 100
    #         tdf = tdf.drop_duplicates(subset=['group'], keep='last')
    #         score = tdf['target_calculation'].tolist()

    #         # score = list(itertools.accumulate(score))

    #         if group == 'left':
    #             line, = axes[n].plot(tdf['uid'], score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         color='k',
    #                         label='New syllable on the left')
    #             lines.append(line)

    #         elif group == 'right':
    #             line, = axes[n].plot(tdf['uid'], score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='gray',
    #                         label='New syllable on the right')
    #             lines.append(line)


    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(4, len(trials)+1, 4))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1
    # # exclude empty axes from fig
    # # axes = [ax for ax in axes if not ax.lines]
    # # for ax in axes:
    # #     fig.delaxes(ax)

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Oral naming (Percent Correct)', ha='center', va='center', fontsize=12, rotation=90)

    # handles = lines
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=4,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_percent_left_right.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()





    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     group_data = cd_data.sort_values(['session', 'uid_in_session'])
    #     trials = [i+1 for i in range(len(group_data))]
    #     # score = group_data['result'].tolist()

    #     tdf = group_data.copy(deep=True)
    #     tdf['uid'] = range(1, len(tdf) + 1)
    #     tdf['group'] = (tdf['uid'] - 1)  // 4
    #     tdf['target_calculation'] = tdf['result'].groupby(tdf['group']).transform('mean') * 100
    #     tdf = tdf.drop_duplicates(subset=['group'], keep='last')
    #     score = tdf['target_calculation'].tolist()

    #     line, = axes[n].plot(tdf['uid'], score,
    #                 linewidth=1,
    #                 linestyle='-',
    #                 color='k',
    #                 label='New syllable on the left')

    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(trials))
    #     axes[n].set_xticks(range(20, len(trials)+1, 20))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1

    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Oral naming (Percent Correct)', ha='center', va='center', fontsize=12, rotation=90)

    # plt.tight_layout()

    # plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_percent_all.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()



    # fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    # fig.set_size_inches(5.27, 10)
    # axes = axes.flatten()

    # n = 0
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     cd_data = cd_data.sort_values(['session', 'uid_in_session'])
    #     cd_data['uid'] = range(1, len(cd_data) + 1)
    #     cd_data['uid_group'] = (cd_data['uid'] - 1)  // 4
    #     cd_data['target_calculation'] = cd_data['result'].groupby(cd_data['uid_group']).transform('mean') * 100

    #     lines = []
    #     for group, group_data in cd_data.groupby('group', sort=False):
    #         if group_data.empty:
    #             raise ValueError(f"No data found for {participant}, {group}")

    #         group_data = group_data.sort_values(['uid'])
    #         score = group_data['target_calculation'].tolist()

    #         if group == 'left':
    #             line, = axes[n].plot(group_data['uid'], score,
    #                         linewidth=1,
    #                         linestyle='-',
    #                         color='k',
    #                         label='New syllable on the left')
    #             lines.append(line)

    #         elif group == 'right':
    #             line, = axes[n].plot(group_data['uid'], score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='gray',
    #                         label='New syllable on the right')
    #             lines.append(line)

    #         elif group == 'none':
    #             line, = axes[n].plot(group_data['uid'], score,
    #                         linewidth=1,
    #                         linestyle=':',
    #                         color='gray',
    #                         label='Teaching syllable on the right')
    #             lines.append(line)


    #     # write participant number
    #     code = f'P{int(participant.split('_')[0]):02d}'
    #     axes[n].text(1.0, 1.0, code, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

    #     # axes[n].set_ylim(0, 1)
    #     # axes[n].set_yticks([0, 0.5, 1])

    #     axes[n].set_xlim(1, len(cd_data['uid']))
    #     axes[n].set_xticks(range(20, len(cd_data['uid'])+1, 20))

    #     axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

    #     axes[n].spines['top'].set_visible(False)
    #     axes[n].spines['right'].set_visible(False)
    #     axes[n].spines['bottom'].set_visible(False)
    #     axes[n].spines['left'].set_visible(False)

    #     n += 1


    # # Add x and y labels in the middle of figure
    # fig.text(0.5, 0.02, 'Trials', ha='center', va='center', fontsize=12)
    # fig.text(0.02, 0.5, 'Oral naming (Percent Correct)', ha='center', va='center', fontsize=12, rotation=90)

    # handles = [lines[2], lines[0], lines[1]]
    # labels = [line.get_label() for line in handles]
    # plt.tight_layout()
    # fig.legend(handles, labels,
    #         loc='upper left',
    #         ncol=2,
    #         columnspacing=1.5,
    #         handletextpad=0.5)

    # plt.subplots_adjust(top=0.90, left=0.12, bottom=0.06, hspace=0.8)
    # # Show the plot
    # extension = 'png'
    # filename = f'isr_all_participants_percent_left_right_teaching.{extension}'
    # print(filename)
    # plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # plt.close()




    # data_dir()
    # cd(os.path.join('analysis', 'output', 'fixations_over_letters_hits'))


    # # normalize session column
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     cd_data = cd_data.sort_values(['session', 'uid_in_session'])
    #     cd_data['uid'] = range(1, len(cd_data) + 1)
    #     # responses = cd_data['response']
    #     fig, axes = plt.subplots(3, 1, figsize=(7, 4), sharex=False, gridspec_kw={'height_ratios': [1, 4, 1]})

    #     ###############################
    #     ########## TOP PLOT ###########
    #     ###############################

    #     tdf = cd_data[cd_data['group'] == 'left']
    #     words = tdf['word']
    #     x = [i+1 for i in range(len(tdf['uid']))]
    #     colors = ['white' if not val else 'black' for val in tdf['result']]

    #     # show x axis on the top
    #     # axes[0].grid(axis='x')
    #     axes[0].set_xticks(x)
    #     axes[0].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=25)
    #     axes[0].set_yticks([1])
    #     axes[0].set_yticklabels(['Oral Naming'])

    #     axes[0].set_xlabel('Words')
    #     axes[0].set_xticklabels(words, rotation=45, ha='right')
    #     axes[0].xaxis.tick_top()
    #     axes[0].xaxis.set_label_position('top')



    #     ###############################
    #     ######### CENTRAL PLOT ########
    #     ###############################

    #     axes[1].set_ylabel('Proportion of fixations\nover syllabels')

    #     lines = []

    #     index_line, = axes[1].plot(x, tdf['score'],
    #                 linewidth=1,
    #                 color='k',
    #                 label='New syllabels on the left')
    #     axes[1].grid(axis='x')
    #     lines.append(index_line)




    #     tdf = cd_data[cd_data['group'] == 'right']
    #     words = tdf['word']

    #     index_line, = axes[1].plot(x, tdf['score'],
    #                 linewidth=1,
    #                 linestyle='--',
    #                 color='k',
    #                 label='New syllabels on the right')
    #     # axes[1].grid(axis='x')
    #     lines.append(index_line)


    #     ###############################
    #     ######### BOTTOM PLOT #########
    #     ###############################


    #     x = [i+1 for i in range(len(tdf['uid']))]
    #     colors = ['white' if not val else 'black' for val in tdf['result']]
    #     # axes[2].grid(axis='x')
    #     axes[2].set_xticks(x)
    #     axes[2].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=25)
    #     axes[2].set_yticks([1])  # Single tick to center labels (optional)
    #     axes[2].set_yticklabels(['Oral Naming'])
    #     # axes[2].set_ylim(0.5, 1.5)
    #     axes[2].set_xlabel('Words')
    #     axes[2].set_xticklabels(words, rotation=45, ha='right')

    #     # get the minimum and maximum
    #     min_val, mid_val, max_val = 0, 0.5, 1
    #     abs_10 = 0.1
    #     yticks = [min_val, mid_val, max_val]
    #     for y in yticks:
    #         axes[1].axhline(y=y, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    #     axes[1].set_ylim([min_val - abs_10, max_val + abs_10])
    #     axes[1].set_yticks([min_val, mid_val, max_val])
    #     axes[1].set_yticklabels([f'Right ({min_val})', f'Both ({mid_val})', f'Left({max_val})'])


    #     for ax in axes:
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.spines['left'].set_visible(False)

    #     # Combine the legend entries from both axes
    #     handles = lines
    #     labels = [line.get_label() for line in handles]

    #     # # Add x and y labels in the middle of figure
    #     fig.text(0.5, -0.02, 'Trials', ha='center', va='center', fontsize=12)

    #     plt.tight_layout()
    #     fig.legend(handles, labels,
    #             loc='upper left',
    #             ncol=4,
    #             columnspacing=1.5,
    #             handletextpad=0.5)

    #     plt.subplots_adjust(top=0.75)

    #     extension = 'png'
    #     filename = f'isr_{participant}_fixations_detailed.{extension}'
    #     print(filename)
    #     plt.savefig(filename, format=extension, dpi=300, transparent=False)
    #     plt.close()



if __name__ == '__main__':
    # plot_accumulated_frequency()
    plot_proportions()
