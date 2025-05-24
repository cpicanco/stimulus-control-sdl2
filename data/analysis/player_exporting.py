import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

from explorer import (
    cache_folder,
    # foldername_1,
    foldername_2,
    # foldername_3,
    load_participant_sources,
    load_pickle,
)

def classify(df, trial, word, section, letter_cluster):
    contained_fixations = df['contained_fixations'].drop('FPOGV', axis=1)
    contained_fixations = contained_fixations.drop('ID', axis=1)
    # rename 'FPOGD' to 'FIXATION_DURATION'
    contained_fixations = contained_fixations.rename(
        columns={
            'FPOGID': 'FIXATION_UID_IN_SESSION',
            'TIME_TICK': 'TIME_TICK_IN_SESSION',
            'FPOGD': 'DURATION',
            'FPOGX': 'X',
            'FPOGY': 'Y'})

    # here we can classify fixations
    contained_fixations['CYCLE'] = trial['cycle']
    contained_fixations['PARTICIPANT'] = f'P{int(str(trial['participant']).split('-')[0]):02d}'

    match = re.match(r'^(\d+)\.data\.processed$', trial['file'])
    session = int(match.group(1))

    contained_fixations['SESSION'] = session
    contained_fixations['SESSION_DATE'] = trial['file_date']
    contained_fixations['SESSION_START_TIME'] = trial['file_start_time']
    contained_fixations['TRIAL_IN_SESSION'] = trial['trial_uid_in_session']
    contained_fixations['WORD'] = word['text']
    contained_fixations['CONDITION'] = trial['condition']
    contained_fixations['RELATION'] = trial['relation']
    contained_fixations['RELATION_CATEGORY'] = word['relation_letter']
    contained_fixations['HAS_DIFFERENTIAL_REINFORCEMENT'] = trial['has_differential_reinforcement']
    contained_fixations['RESULT'] = trial['result']
    response : str
    if trial['relation'] == 'CD':
        response = trial['response'].strip().replace('-4', '')
    else:
        response = trial['response'].strip()
    contained_fixations['RESPONSE'] = response
    contained_fixations['LATENCY'] = trial['latency']
    contained_fixations['SECTION'] = section
    contained_fixations['LETTER_CLUSTER'] = letter_cluster

    return contained_fixations

def plot_cluster(df, classifier, target_column):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    df['LETTER_COLOR'] = [f'C{color}' for color in df[target_column]]

    ax.scatter(df['X'], df['Y'], c=df['LETTER_COLOR'], s=1)
    if classifier is not None:
        scatter = ax.scatter(classifier.cluster_centers_[:, 0], classifier.cluster_centers_[:, 1],
                    c='k',  marker='x', s=100, label='Centroids')
    else:
        scatter = None
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1440)
    ax.set_ylim(0, 900)

    unique_colors = df['LETTER_COLOR'].unique()
    handles = []
    labels = []

    if scatter is not None:
        handles.append(scatter)
        labels.append(scatter.get_label())

    color_label_map = {n: f'Cluster {i+1}' for i, n in enumerate(sorted(unique_colors))}

    for color, label in color_label_map.items():
        # Create a proxy artist for the legend
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=5))
        labels.append(label)

    # Add the legend
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_line(df, column):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df[column])
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_histogram(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, result in zip(axes, [0, 1]):
        cd_probes = df[(df['result'] == result)].copy()

        # Split into left/right
        left = cd_probes[cd_probes['group'] == 'left']
        right = cd_probes[cd_probes['group'] == 'right']

        sns.histplot(data=left, x='score_delta', bins=20, kde=True, color='red', alpha=0.5, label='Left', ax=ax)
        sns.histplot(data=right, x='score_delta', bins=20, kde=True, color='blue', alpha=0.5, label='Right', ax=ax)

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
        mean_left = left['score_delta'].mean()
        mean_right = right['score_delta'].mean()
        ax.axvline(x=mean_left, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=mean_right, color='blue', linestyle='--', alpha=0.5)

        # Add statistical annotation
        text = (f'Left Mean (N={len(left)})= {mean_left:.2f}     Right Mean (N={len(right)})= {mean_right:.2f}')
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


def export_CD():
    # load sources
    sources = load_participant_sources(foldername_2)
    kmeans = KMeans(n_clusters=4, random_state=42)
    # loop over participants
    participants = []
    for participant in sources:
        dfs = []
        participant_metadata = sources[participant].pop('metadata')
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    filename = '_'.join([
                        participant_metadata['study'],
                        participant.replace('-', '_'), source['code']])
                    print(filename)

                    trials = load_pickle(filename)

                    for trial in trials:
                        if (trial['relation'] == 'CD'):
                            prefix = 'sample'
                            if trial[f'{prefix}_fixations'] is not None:
                                fixations_by_word, word_measures = trial[f'{prefix}_fixations']
                                for i, word in enumerate(fixations_by_word):
                                    # drop 'FPOGV'
                                    contained_fixations = word['contained_fixations'].drop('FPOGV', axis=1)
                                    if contained_fixations.empty:
                                        print(f'Warning: empty word for trial {participant} {trial['file']} {trial["trial_uid_in_session"]}')
                                        continue
                                    contained_fixations = contained_fixations.drop('ID', axis=1)
                                    # rename 'FPOGD' to 'FIXATION_DURATION'
                                    contained_fixations = contained_fixations.rename(
                                        columns={
                                            'FPOGID': 'FIXATION_UID_IN_SESSION',
                                            'TIME_TICK': 'TIME_TICK_IN_SESSION',
                                            'FPOGD': 'DURATION',
                                            'FPOGX': 'X',
                                            'FPOGY': 'Y'})

                                    # here we can classify fixations
                                    contained_fixations['CYCLE'] = trial['cycle']
                                    contained_fixations['PARTICIPANT'] = f'P{int(str(trial['participant']).split('-')[0]):02d}'

                                    match = re.match(r'^(\d+)\.data\.processed$', trial['file'])
                                    if match:
                                        session = int(match.group(1))

                                    contained_fixations['SESSION'] = session
                                    contained_fixations['SESSION_DATE'] = trial['file_date']
                                    contained_fixations['SESSION_START_TIME'] = trial['file_start_time']
                                    contained_fixations['TRIAL_IN_SESSION'] = trial['trial_uid_in_session']
                                    contained_fixations['WORD'] = word['text']
                                    contained_fixations['CONDITION'] = trial['condition']
                                    contained_fixations['RELATION'] = trial['relation']
                                    contained_fixations['RELATION_CATEGORY'] = word['relation_letter']
                                    contained_fixations['HAS_DIFFERENTIAL_REINFORCEMENT'] = trial['has_differential_reinforcement']
                                    contained_fixations['RESULT'] = trial['result']
                                    contained_fixations['RESPONSE'] = trial['response'].strip()
                                    contained_fixations['LATENCY'] = trial['latency']

                                    dfs.append(contained_fixations)

        df = pd.concat(dfs, ignore_index=True)
        df['LETTER_CLUSTER'] = kmeans.fit_predict(df[['X', 'Y']])
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0])  # Sorts left-to-right
        label_mapping = {original: new_label+1 for new_label, original in enumerate(sorted_indices)}
        df['LETTER_CLUSTER'] = df['LETTER_CLUSTER'].map(label_mapping)
        participants.append(df)
        # plot_cluster(df, kmeans)

    exporting = pd.concat(participants, ignore_index=True)
    exporting['RESPONSE'] = exporting['RESPONSE'].str.replace('-4', '')
    # export to csv
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_fixations_CD.csv')
    exporting.to_csv(filename, index=False)


def export_all_relations():
    sources = load_participant_sources(foldername_2)

    dfs = []
    for participant in sources:
        participant_metadata = sources[participant].pop('metadata')
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    filename = '_'.join([
                        participant_metadata['study'],
                        participant.replace('-', '_'), source['code']])
                    print(filename)

                    trials = load_pickle(filename)

                    for trial in trials:
                        for prefix in ['sample', 'comparisons']:
                            if trial[f'{prefix}_fixations'] is not None:
                                fixations_by_word, _ = trial[f'{prefix}_fixations']
                                for i, word in enumerate(fixations_by_word):
                                    if 'letter_fixations' in word:
                                        fixations_by_letter, _ = word['letter_fixations']
                                        for j, letter in enumerate(fixations_by_letter):
                                            if letter['contained_fixations'].empty:
                                                continue
                                            dfs.append(classify(letter, trial, word, prefix, j+1))
                                    else:
                                        dfs.append(classify(word, trial, word, prefix, i+5))

        # df = pd.concat(dfs, ignore_index=True)
        # target_column = 'LETTER_CLUSTER'
        # df[target_column] = kmeans.fit_predict(df[['X', 'Y']])
        # centroids = kmeans.cluster_centers_
        # sorted_indices = np.argsort(centroids[:, 0])  # Sorts left-to-right
        # label_mapping = {original: new_label+1 for new_label, original in enumerate(sorted_indices)}
        # df[target_column] = df[target_column].map(label_mapping)
        # plot_cluster(df, None, target_column)

    exporting = pd.concat(dfs, ignore_index=True)
    # use up to 4 decimal places
    exporting['X'] = exporting['X'].round(4)
    exporting['Y'] = exporting['Y'].round(4)
    exporting['DURATION'] = exporting['DURATION'].round(4)
    exporting['LATENCY'] = exporting['LATENCY'].round(4)
    exporting['TIME_TICK_IN_SESSION'] = exporting['TIME_TICK_IN_SESSION'].round(4)

    # sort by participant, date, and timetick in session
    exporting = exporting.sort_values(by=['PARTICIPANT', 'SESSION_DATE', 'TIME_TICK_IN_SESSION'])

    # export to csv
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_fixations_all_relations.csv')
    exporting.to_csv(filename, index=False)



def export_b():
    # load sources
    sources = load_participant_sources(foldername_2)

    # loop over participants
    dfs = []
    for participant in sources:
        participant_metadata = sources[participant].pop('metadata')
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    filename = '_'.join([
                        participant_metadata['study'],
                        participant.replace('-', '_'), source['code']])
                    print(filename)

                    trials = load_pickle(filename+'_raw_fixations')

                    for trial in trials:
                        prefixes = ['sample', 'comparisons']
                        for prefix in prefixes:
                            # if not events = trial['events']['relation'] = 'AB':
                            #     continue

                            if trial['sections'][prefix] is not None:
                                fixations = trial['sections'][prefix]['fixations']
                                relation_letter = trial['sections'][prefix]['relation_letter']
                                events = trial['events']

                                if fixations.empty:
                                    print(f'Warning: empty fixations for {participant}')
                                    continue

                                fixations = fixations.rename(
                                    columns={
                                        'FPOGID': 'FIXATION_UID_IN_SESSION',
                                        'TIME_TICK': 'TIME_TICK_IN_SESSION',
                                        'FPOGD': 'DURATION',
                                        'FPOGX': 'X',
                                        'FPOGY': 'Y',
                                        'FPOGV': 'VALID'})

                                # here we can classify fixations
                                fixations['CYCLE'] = events['Cycle.ID']
                                fixations['PARTICIPANT'] = f'P{int(str(events['Participant']).split('-')[0]):02d}'

                                match = re.match(r'^(\d+)\.data\.processed$', events['File'])
                                if match:
                                    session = int(match.group(1))

                                fixations['SESSION'] = session
                                fixations['SESSION_DATE'] = events['Date']
                                fixations['SESSION_START_TIME'] = events['Time']
                                fixations['TRIAL_IN_SESSION'] = events['Session.Trial.UID']
                                fixations['WORD'] = events['Name']
                                fixations['CONDITION'] = events['Condition']
                                fixations['RELATION'] = events['Relation']
                                fixations['RELATION_CATEGORY'] = relation_letter
                                fixations['HAS_DIFFERENTIAL_REINFORCEMENT'] = events['HasDifferentialReinforcement']
                                fixations['RESULT'] = events['Result']
                                fixations['RESPONSE'] = events['Response'].strip()
                                fixations['LATENCY'] = events['Latency']
                                fixations['SECTION'] = prefix

                                dfs.append(fixations)

    exporting = pd.concat(dfs)

    # export to csv
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_raw_fixations.csv')
    exporting.to_csv(filename, index=False)

def plot_clusters():
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_raw_fixations.csv')
    dfs = pd.read_csv(filename)
    dfs.to_pickle(os.path.join(folder, f'{foldername_2}_raw_fixations.pkl'))

    # # load classifier
    # kmeans = KMeans(n_clusters=4, random_state=42)

    # for participant, df in dfs.groupby('PARTICIPANT'):
    #     target_column = 'POSITION_CLUSTER'
    #     df[target_column] = kmeans.fit_predict(df[['X', 'Y']])
    #     centroids = kmeans.cluster_centers_
    #     sorted_indices = np.argsort(centroids[:, 0])  # Sorts left-to-right
    #     label_mapping = {original: new_label for new_label, original in enumerate(sorted_indices)}
    #     df[target_column] = df[target_column].map(label_mapping)
    #     plot_cluster(df, kmeans, target_column)


def export_cluster_stats():
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_fixations.csv')
    df = pd.read_csv(filename)

    # Calculate cluster statistics using groupby and unstack
    cluster_stats = (
        df.groupby(['PARTICIPANT', 'SESSION', 'TRIAL_IN_SESSION', 'LETTER_CLUSTER'])
        .agg(
            count=('DURATION', 'size'),
            duration=('DURATION', 'sum')
        )
        .unstack(fill_value=0)
    )

    # Flatten multi-level columns
    cluster_stats.columns = [
        f'letter{int(cluster+1)}_count' if stat == 'count' else f'letter{int(cluster+1)}_duration'
        for (stat, cluster) in cluster_stats.columns
    ]
    cluster_stats = cluster_stats.reset_index()

    # Aggregate other trial-level metadata
    trial_metadata = (
        df.groupby(['PARTICIPANT', 'SESSION', 'TRIAL_IN_SESSION'])
        .agg({
            'SESSION_DATE': 'first',
            'SESSION_START_TIME': 'first',
            'CYCLE': 'first',
            'WORD': 'first',
            'CONDITION': 'first',
            'RELATION': 'first',
            'RELATION_CATEGORY': 'first',
            'RESULT': 'first',
            'RESPONSE': 'first',
            'LATENCY': 'first',
            'HAS_DIFFERENTIAL_REINFORCEMENT': 'first',
        })
        .reset_index()
    )

    # Merge cluster stats with metadata
    result_df = pd.merge(
        cluster_stats,
        trial_metadata,
        on=['PARTICIPANT', 'SESSION', 'TRIAL_IN_SESSION']
    )

    # Rename columns to match target schema
    result_df = result_df.rename(columns={
        'SESSION_DATE': 'session_date',
        'SESSION_START_TIME': 'session_start_time',
        'PARTICIPANT': 'participant',
        'SESSION': 'session',
        'TRIAL_IN_SESSION': 'trial_in_session',
        'CYCLE': 'cycle',
        'WORD': 'word',
        'CONDITION': 'condition',
        'RELATION': 'relation',
        'RELATION_CATEGORY': 'relation_category',
        'RESULT': 'result',
        'RESPONSE': 'response',
        'LATENCY': 'latency',
        'HAS_DIFFERENTIAL_REINFORCEMENT': 'has_differential_reinforcement'
    })

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')
    result_df.to_csv(filename, index=False)


def calculate_index(row,
                    col1='letter1_duration',
                    col2='letter2_duration',
                    col3='letter3_duration',
                    col4='letter4_duration',
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

def cd(directory, verbose=False):
    os.chdir(directory)
    if verbose:
        print("Current Working Directory: ", os.getcwd())

def data_dir(verbose=False):
    if os.getcwd().endswith('data'):
        return
    else:
        cd('..', verbose)
        data_dir(verbose)

if __name__ == '__main__':
    export_all_relations()
    # plot_clusters()
    # export_cluster_stats()
    
    
    
    
    # from correlation import plot_correlation
    # from words import (
    #     isr_left,
    #     isr_both,
    #     isr_right,
    #     teaching
    # )

    # data_dir()
    # cd(os.path.join('analysis', 'output', 'fixations_over_letters_v3'))

    # folder = cache_folder()
    # filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')

    # df = pd.read_csv(filename)

    # df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)
    # df.loc[:, 'score'] = df.apply(calculate_index, axis=1, use_delta=False)
    # df.loc[:, 'score_delta'] = df.apply(calculate_index, axis=1, use_delta=True)
    # df.loc[:, 'session'] = (df.groupby('participant')['session']
    #                 .rank(method='dense')
    #                 .astype(int))
    # max_sessions = df.groupby('participant')['session'].transform('max')
    # df.loc[df['session'] == max_sessions, 'cycle'] = 7

    # word_filters = [
    #     df['word'].str.match(isr_left),
    #     df['word'].str.match(isr_right),
    #     df['word'].str.match(isr_both),
    #     df['word'].str.match(teaching)]

    # groups = [
    #     'left',
    #     'right',
    #     'both',
    #     'none'
    # ]

    # df['group'] = np.select(word_filters, groups, default='other')

    # df = df[df['condition'] == '7']
    # plot_histogram(df)


    # hits_df = df.pivot_table(
    #     index='participant',
    #     columns='group',
    #     values='result',
    #     aggfunc='sum'
    # )

    # mean_df = df.pivot_table(
    #     index='participant',
    #     columns='group',
    #     values='score',
    #     aggfunc='mean'
    # )

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

    # delta_df = pd.concat([mean_df, count_df, hits_df, mean_delta_df], axis=1)
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

    #         group_data = group_data.sort_values(['session', 'trial_in_session'])

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
    #             lines.append(line)

    #         elif group == 'right':
    #             line, = axes[n].plot(trials, score,
    #                         linewidth=1,
    #                         linestyle='--',
    #                         color='k',
    #                         label='New syllable on the right')
    #             lines.append(line)


    #     # write participant number
    #     axes[n].text(1.0, 1.0, participant, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

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

    # handles = [lines[0], lines[1]]
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

    #     cd_data = cd_data.sort_values(['session', 'trial_in_session'])
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
    #                         label='Teaching syllables')
    #             lines.append(line)


    #     # write participant number
    #     axes[n].text(1.0, 1.0, participant, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

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








    # # data_dir()
    # # cd(os.path.join('analysis', 'output', 'fixations_over_letters_hits'))


    # # # normalize session column
    # # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    # #     print(f"Participant: {participant} Difference: {difference}")

    # #     cd_data = cd_data.sort_values(['session', 'trial_in_session'])
    # #     cd_data['uid'] = range(1, len(cd_data) + 1)
    # #     # responses = cd_data['response']
    # #     fig, axes = plt.subplots(3, 1, figsize=(7, 4), sharex=False, gridspec_kw={'height_ratios': [1, 4, 1]})

    # #     ###############################
    # #     ########## TOP PLOT ###########
    # #     ###############################

    # #     tdf = cd_data[cd_data['group'] == 'left']
    # #     words = tdf['reference_word']
    # #     x = [i+1 for i in range(len(tdf['uid']))]
    # #     colors = ['white' if not val else 'black' for val in tdf['result']]

    # #     # show x axis on the top
    # #     # axes[0].grid(axis='x')
    # #     axes[0].set_xticks(x)
    # #     axes[0].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=25)
    # #     axes[0].set_yticks([1])
    # #     axes[0].set_yticklabels(['Oral Naming'])

    # #     axes[0].set_xlabel('Words')
    # #     axes[0].set_xticklabels(words, rotation=45, ha='right')
    # #     axes[0].xaxis.tick_top()
    # #     axes[0].xaxis.set_label_position('top')



    # #     ###############################
    # #     ######### CENTRAL PLOT ########
    # #     ###############################

    # #     axes[1].set_ylabel('Proportion of fixations\nover syllabels')

    # #     lines = []

    # #     index_line, = axes[1].plot(x, tdf['score'],
    # #                 linewidth=1,
    # #                 color='k',
    # #                 label='New syllabels on the left')
    # #     axes[1].grid(axis='x')
    # #     lines.append(index_line)




    # #     tdf = cd_data[cd_data['group'] == 'right']
    # #     words = tdf['reference_word']

    # #     index_line, = axes[1].plot(x, tdf['score'],
    # #                 linewidth=1,
    # #                 linestyle='--',
    # #                 color='k',
    # #                 label='New syllabels on the right')
    # #     # axes[1].grid(axis='x')
    # #     lines.append(index_line)


    # #     ###############################
    # #     ######### BOTTOM PLOT #########
    # #     ###############################


    # #     x = [i+1 for i in range(len(tdf['uid']))]
    # #     colors = ['white' if not val else 'black' for val in tdf['result']]
    # #     # axes[2].grid(axis='x')
    # #     axes[2].set_xticks(x)
    # #     axes[2].scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=25)
    # #     axes[2].set_yticks([1])  # Single tick to center labels (optional)
    # #     axes[2].set_yticklabels(['Oral Naming'])
    # #     # axes[2].set_ylim(0.5, 1.5)
    # #     axes[2].set_xlabel('Words')
    # #     axes[2].set_xticklabels(words, rotation=45, ha='right')

    # #     # get the minimum and maximum
    # #     min_val, mid_val, max_val = 0, 0.5, 1
    # #     abs_10 = 0.1
    # #     yticks = [min_val, mid_val, max_val]
    # #     for y in yticks:
    # #         axes[1].axhline(y=y, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    # #     axes[1].set_ylim([min_val - abs_10, max_val + abs_10])
    # #     axes[1].set_yticks([min_val, mid_val, max_val])
    # #     axes[1].set_yticklabels([f'Right ({min_val})', f'Both ({mid_val})', f'Left({max_val})'])


    # #     for ax in axes:
    # #         ax.spines['top'].set_visible(False)
    # #         ax.spines['right'].set_visible(False)
    # #         ax.spines['bottom'].set_visible(False)
    # #         ax.spines['left'].set_visible(False)

    # #     # Combine the legend entries from both axes
    # #     handles = lines
    # #     labels = [line.get_label() for line in handles]

    # #     # # Add x and y labels in the middle of figure
    # #     fig.text(0.5, -0.02, 'Trials', ha='center', va='center', fontsize=12)

    # #     plt.tight_layout()
    # #     fig.legend(handles, labels,
    # #             loc='upper left',
    # #             ncol=4,
    # #             columnspacing=1.5,
    # #             handletextpad=0.5)

    # #     plt.subplots_adjust(top=0.75)

    # #     extension = 'png'
    # #     filename = f'isr_{participant}_fixations_detailed.{extension}'
    # #     print(filename)
    # #     plt.savefig(filename, format=extension, dpi=300, transparent=False)
    # #     plt.close()
