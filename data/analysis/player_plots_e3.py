from itertools import pairwise

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np

from player_utils import calculate
from fileutils import data_dir, cd

def pairwise_non_overlapping(iterable):
    it = iter(iterable)
    return list(zip(it, it))

marker1 = 'd'
marker2 = 'd_'
marker3 = 'o'
marker4 = 'o_'

marker_label_map = {
    marker1: 'teaching',
    marker2: 'assessment (teaching words)',
    marker3: 'assessment (generalization words)',
    marker4: 'assessment (MPP)'
}

def by_trial_subplots(data: pd.DataFrame, measure: str, calculation: str):
    def draw(ax: plt.Axes,
             group: pd.DataFrame,
             vertical_lines: list,
             use_y_limit: bool,
             maximum_y: float,
             minimum_y: float):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['target_calculation'],
                marker=marker.replace('_', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, zorder=0)

        # draw horizontal line for 0%, 50% of trials, 100%
        if use_y_limit:
            yticks = [0, 50, 100]
            for y in yticks:
                ax.axhline(y=y, color='black', linestyle='--',  linewidth=0.5, zorder=0)
            ax.set_ylim([-20, 120])
            ax.set_yticks(yticks)
        else:
            # get the minimum and maximum
            min_val, max_val = minimum_y, maximum_y
            mid_val = (max_val + min_val) / 2
            # get absolute 10% of the range to  subtract from min and add to max
            abs_10 = (max_val - min_val) * 0.1
            yticks = [min_val, mid_val, max_val]
            for y in yticks:
                ax.axhline(y=y, color='black', linestyle='--', linewidth=0.5, zorder=0)
            ax.set_ylim([min_val - abs_10, max_val + abs_10])
            ax.set_yticks([min_val, mid_val, max_val])


    number_of_rows = len(data['relation'].unique())
    unique_sessions = data['file'].unique()

    figsize = (7, 8)
    fig, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=figsize,
        sharex=False, sharey=True)
    mapping = {
        'AB' : {'subtitle':'S: word sound\nC: pictures', 'vertical_lines': None},
        'AC' : {'subtitle':'S: word sound\nC: written words', 'vertical_lines': None},
        'BC' : {'subtitle':'S: picture\nC: written words', 'vertical_lines': None},
        'CB' : {'subtitle':'S: written word\nC: pictures', 'vertical_lines': None},
        'CD' : {'subtitle':'Oral reading/\nTextual behavior', 'vertical_lines': None}
    }
    fontsize = 10

    # get minimum and maximum
    minimum_y = 0
    maximum_y = 0
    for idx, (relation, group) in enumerate(data.groupby('relation')):
        # reset uid
        group['uid'] = range(1, len(group) + 1)

        processed_group = []
        for (relation, condition, session), g in group.groupby(['relation', 'condition', 'session']):
            if measure == 'hits':
                target_column = 'result'
                percentage = True
                y_limit = True
                ylabel = f'Percent correct (average)'

            elif measure == 'latency':
                target_column = 'latency'
                percentage = False
                y_limit = False
                ylabel = f'Latency (average)'

            d = calculate(g, calculation=calculation, target_column=target_column, percentage=percentage)
            processed_group.append(d)

        group = pd.concat(processed_group, axis=0)
        minimum_y = np.min([minimum_y, group['target_calculation'].min()])
        maximum_y = np.max([maximum_y, group['target_calculation'].max()])

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        # reset uid
        group['uid'] = range(1, len(group) + 1)

        processed_group = []
        for (relation, condition, session), g in group.groupby(['relation', 'condition', 'session']):
            if measure == 'hits':
                target_column = 'result'
                percentage = True
                y_limit = True
                ylabel = f'Percent correct (average)'

            elif measure == 'latency':
                target_column = 'latency'
                percentage = False
                y_limit = False
                ylabel = f'Latency (average)'

            d = calculate(g, calculation=calculation, target_column=target_column, percentage=percentage)
            processed_group.append(d)

        group = pd.concat(processed_group, axis=0)

        vertical_lines = []
        cycles_onset = []
        if relation ==  'CD':
            for (cycle, condition, session), g in group.groupby(['cycle', 'condition', 'file']):
                # print(cycle, condition, session)
                if condition == '7':
                    vertical_lines.append(g['uid'].values[0])
                    vertical_lines.append(g['uid'].values[-1])

                    if session == unique_sessions[-1]:
                        continue
                    else:
                        cycles_onset.append(g['uid'].values[-1])

            cycles_onset.insert(0, vertical_lines[0])
            cycles_onset.append(group['uid'].max())
            axes[idx].set_xticks(cycles_onset)
            # remove first and last vertical line
            vertical_lines.sort()
            mapping[relation]['vertical_lines'] = vertical_lines

        if relation == 'AB':
            for (cycle, condition, session), g in group.groupby(['cycle', 'condition', 'session']):
                if condition == '1':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())

            axes[idx].set_xticks(vertical_lines)
            mapping[relation]['vertical_lines'] = vertical_lines

        if relation ==  'AC':
            for (cycle, condition, session), g in group.groupby(['cycle', 'condition', 'session']):
                if condition == '2a':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())
            axes[idx].set_xticks(vertical_lines)
            mapping[relation]['vertical_lines'] = vertical_lines

        if (relation ==  'BC') or (relation == 'CB'):
            for (cycle, condition, session), g in group.groupby(['cycle', 'condition', 'session']):
                if condition == '3':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())

            axes[idx].set_xticks(vertical_lines)
            mapping[relation]['vertical_lines'] = vertical_lines

        draw(ax=axes[idx],
             group=group,
             vertical_lines=vertical_lines[1:-1],
             use_y_limit=y_limit,
             maximum_y=maximum_y,
             minimum_y=minimum_y)

    axes[-1].set_xlabel('Trials (groups of 4)')
    # axes[2].set_ylabel(ylabel, rotation=0, ha='left', va='top')

    participant = data['participant'].iloc[0]

    fig.text(0.02, 0.02, participant, ha='left', va='bottom', fontsize=12)

    labels = [
        'teaching',
        'assessment (teaching words)',
        'assessment (generalization words)',
        'assessment (multiple probes procedure)'
    ]

    handles = []
    handles = [
        Line2D([0], [0],
            marker=marker1.replace('_', ''),
            color='black',
            markerfacecolor='white',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker1]),
        Line2D([0], [0],
            marker=marker2.replace('_', ''),
            color='black',
            markerfacecolor='black',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker2]),
        Line2D([0], [0],
            marker=marker3.replace('_', ''),
            color='black',
            markerfacecolor='none',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker3]),
        Line2D([0], [0],
            marker=marker4.replace('_', ''),
            color='black',
            markerfacecolor='black',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker4]),
    ]

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        axes[idx].set_ylabel(
            mapping[relation]['subtitle'],
            fontsize=fontsize,
            rotation=0,
            ha='left',
            va='top')
        axes[idx].yaxis.set_label_position("right")
        axes[idx].yaxis.set_label_coords(1.025, 0.90)

    plt.tight_layout()

    top_padding = 0.88
    left_padding = 0.15
    bottom_padding = 0.2

    fig.legend(
        handles,
        labels,
        ncol=2,
        fontsize=fontsize,
        loc='upper center')
    fig.suptitle(ylabel, y=0.5, x=0.05, va='center', fontsize=fontsize, rotation=90)
    plt.subplots_adjust(top=top_padding, left=left_padding, bottom=bottom_padding)

    top_subplot_padding = 0.05
    for idx, (relation, group) in enumerate(data.groupby('relation')):
        ax = axes[idx]
        pos = ax.get_position()
        if idx == 2 or idx == 3:
            ax.set_position([pos.x0, pos.y0 - top_subplot_padding, pos.width, pos.height - (pos.height*0.1)])  # Reduce height to add space

        elif idx == 4:
            ax.set_position([pos.x0, pos.y0 - top_subplot_padding*2.8, pos.width, pos.height - (pos.height*0.1)])  # Reduce height to add space

        else:
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height - (pos.height*0.1)])

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        ax = axes[idx]
        bbox = ax.get_position()
        top = (bbox.ymax) + (bbox.ymax)*0.025
        vertical_lines = mapping[relation]['vertical_lines']
        # Create a transform that uses data coords for x and figure coords for y
        transform = mtransforms.blended_transform_factory(ax.transData, fig.transFigure)
        if idx == 0:
            for i, (onset, offset) in enumerate(pairwise(vertical_lines)):
                midpoint = (onset + offset) / 2  # Calculate midpoint in data coordinates
                # Place text using blended transform (data x, figure y)
                fig.text(midpoint, top, f'C{i+1}', ha='center', va='center',
                        fontsize=fontsize, transform=transform, rotation=45)
        elif idx == 4:
            top = (bbox.ymax) + (bbox.ymax)*0.2
            midpoint_adjust = 0.03
            for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
                midpoint = (onset + offset) / 2  # Calculate midpoint in data coordinates
                midpoint = midpoint + (midpoint * midpoint_adjust)
                # Place text using blended transform (data x, figure y)
                fig.text(midpoint, top, f'P{i+1}', ha='center', va='center',
                        fontsize=fontsize, transform=transform, rotation=45)

            for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines[1:])):
                midpoint = (onset + offset) / 2  # Calculate midpoint in data coordinates
                midpoint = midpoint + (midpoint * midpoint_adjust)
                # Place text using blended transform (data x, figure y)
                fig.text(midpoint, top, f'C{i+1}', ha='center', va='center',
                        fontsize=fontsize, transform=transform, rotation=45)



    # Show the plot
    filename = f'{participant}_{measure}_by_trial_{calculation}.pdf'
    print(filename)
    plt.savefig(filename, format='pdf', dpi=300, transparent=False)
    plt.close()

if __name__ == '__main__':
    import os
    from explorer import (
        foldername_3,
        load_from_csv,
        export
    )

    df = load_from_csv(foldername_3)

    data_dir()
    cd(os.path.join('analysis', 'output', foldername_3, 'figures'))

    color_map = {
        '1': 'black',
        '2a': 'black',
        '2b': 'black',
        '3': 'black',
        '4': 'black',
        '5': 'black',
        '6': 'black',
        '7': 'black'
    }
    df['edgecolors'] = df['condition'].map(color_map)

    facecolor_map = {
        '1': 'white',
        '2a': 'white',
        '2b': 'white',
        '3': 'black',
        '4': 'white',
        '5': 'white',
        '6': 'white',
        '7': 'black'
    }
    df['facecolors'] = df['condition'].map(facecolor_map)

    marker_map = {
        '1': marker1,
        '2a': marker1,
        '2b': marker1,
        '3': marker2,
        '4': marker3,
        '5': marker3,
        '6': marker3,
        '7': marker3
    }
    df['marker'] = df['condition'].map(marker_map)
    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)

    # normalize session column
    i = 0
    for participant, participant_data in df.groupby('participant'):
        participant_data['file_num'] = participant_data['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        unique_sorted = participant_data['file_num'].sort_values().unique()
        mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        participant_data['session'] = participant_data['file_num'].map(mapping) + 1
        # export(participant_data, '_teste.csv')

        by_trial_subplots(participant_data,
                          measure='hits', calculation='rolling_average_no_overlap')

        by_trial_subplots(participant_data,
                          measure='latency', calculation='rolling_average_no_overlap')

    cd(os.path.join('..', '..'))