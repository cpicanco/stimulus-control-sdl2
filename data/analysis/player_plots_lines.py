import os

from fileutils import (
    cd,
    # walk_and_execute
)
# from metadata_calculator import collect_metadata

from barplots import (
    # dispersion_plot_per_cycle,
    # default_axis_config,
    save_or_show
)

import numpy as np
import matplotlib.pyplot as plt

from words import (
    nibo, fale, bofa, leni, lebo, fani, boni, lefa, fabo, nile, bole, nifa,
    bona, lefi, fabe, nilo, lani, febo, nole, bifa)

from speech import load_probes_file

def multiple_probes_procedure_lines(CD_probes, participants_list, plot_type='line', group=False, append_to_filename=''):
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

    w11_filter = r'(nibe|lofi)'
    w12_filter = r'(bofi|nale)'
    w13_filter = r'(leba|nofa)'
    w14_filter = r'(febi|lano)'
    w15_filter = r'(bena|falo)'


    filters = [
        w1_filter, w2_filter, w3_filter, w4_filter, w5_filter,
        w6_filter, w7_filter, w8_filter, w9_filter, w10_filter,
        w11_filter, w12_filter, w13_filter, w14_filter, w15_filter
        ]

    CD_probes = CD_probes[(CD_probes['Condition'] == 7) | (CD_probes['Condition'] == 8)]

    # replace all '\' to '' in Participant column
    CD_probes.loc[:, 'Participant'] = CD_probes['Participant'].str.replace('\\', '')

    # need to treat some exceptions
    # must join Files "002.data.processed" and "006.data.processed" ONLY for for Participant "13-AND"
    CD_probes.loc[(CD_probes['Participant'] == '13-AND') & (CD_probes['File'] == '006.data.processed'), 'File'] = '002.data.processed'

    # must join Files "049.data.processed" and "050.data.processed" ONLY for for Participant "20-CAM"
    CD_probes.loc[(CD_probes['Participant'] == '20-CAM') & (CD_probes['File'] == '050.data.processed'), 'File'] = '049.data.processed'

    # must join Files "034.data.processed" and "037.data.processed" ONLY for for Participant "3-PVV "
    CD_probes.loc[(CD_probes['Participant'] == '3-PVV') & (CD_probes['File'] == '037.data.processed'), 'File'] = '034.data.processed'

    # must join Files "017.data.processed" and "022.data.processed" ONLY for for Participant "11-BLI24"
    CD_probes.loc[(CD_probes['Participant'] == '11-BLI24') & (CD_probes['File'] == '022.data.processed'), 'File'] = '017.data.processed'


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
                if filter not in data_to_plot[file_tag]:
                    data_to_plot[file_tag][filter] = []

                word_probes = file_probes[file_probes['Name'].str.match(filter)]
                if word_probes.empty:
                    data_to_plot[file_tag][filter].append(np.nan)

                else:
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
            # x tick labels
            ax.set_xticklabels(['*', '1', '2', '3', '4', '5', '6'])

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

    save_or_show(fig, True, f'{append_to_filename}_line_per_cycle_CD_probes.pdf')


if __name__ == '__main__':
    from study3_constants import (
        participants,
        foldername)

    cd(os.path.join('output', foldername))
    filename = 'probes_CD.data.processed'
    CD_probes = load_probes_file(filename)

    cd('figures')

    # multiple_probes_procedure(
    #     participants_list=participant_folders,
    #     group=True,
    #     append_to_filename='_all')

    for participant in participants:
        multiple_probes_procedure_lines(
            CD_probes,
            participants_list=[participant],
            group=False,
            append_to_filename=f'{participant}')

    # line_plot_AB_AC_CD()
    # data = get_hits_trials_per_participant_per_cycle_per_phase(participants_list=participant_folders)
    # paired_chi_squared(data)
    # count_min_max(participants_list=participant_folders)
    # plot(participants_list=participant_folders)
    # count_errors()
    # plot_rank_tests()

    cd(os.path.join('..', '..', '..'))