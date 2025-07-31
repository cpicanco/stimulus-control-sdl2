import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

from explorer import (
    cache_folder,
    foldername_2,
    load_participant_sources,
    load_pickle,
)


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

def plot_clusters():
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_raw_fixations.csv')
    dfs = pd.read_csv(filename)
    dfs = dfs[dfs['RELATION'] == 'CD']
    dfs = dfs[dfs['SECTION'] == 'sample']
    # dfs.to_pickle(os.path.join(folder, f'{foldername_2}_raw_fixations.pkl'))

    # # load classifier
    kmeans = KMeans(n_clusters=4, random_state=42)

    for participant, df in dfs.groupby('PARTICIPANT'):
        target_column = 'POSITION_CLUSTER'
        df[target_column] = kmeans.fit_predict(df[['X', 'Y']])
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0])  # Sorts left-to-right
        label_mapping = {original: new_label for new_label, original in enumerate(sorted_indices)}
        df[target_column] = df[target_column].map(label_mapping)
        plot_cluster(df, kmeans, target_column)


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

        sns.histplot(data=left, x='score_delta', bins=20, kde=True, color='red', alpha=0.5, label='Esquerda', ax=ax)
        sns.histplot(data=right, x='score_delta', bins=20, kde=True, color='blue', alpha=0.5, label='Direita', ax=ax)

        ax.set_xlim(1.1, -1.1)
        ax.set_xlabel('Olhou mais para a esquerda | Proporção | Olhou mais para a direita')
        if result == 0:
            text = 'Erros'
        else:
            text = 'Acertos'

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
        text = (f'Média Esquerda (N={len(left)})= {mean_left:.2f}     Média Direita (N={len(right)})= {mean_right:.2f}')
        ax.text(x=1.0, y=1.13, s=text,
                ha='right', va='top', transform=ax.transAxes)

    ax.legend(
        title='Posição da sílaba nova',
        labels=['Esquerda', 'Direita'],
        loc='upper right',
        ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    extension = 'svg'
    filename = f'isr_fixations_over_positions_density_pt_br.{extension}'
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
                        participant.replace('-', '_'), source['code'],
                        'processed_fixations'])
                    print(filename)

                    trials = load_pickle(filename)

                    for trial in trials:
                        tdf = trial['data']
                        if (tdf['Relation'] == 'CD'):
                            prefix = 'sample'
                            if trial[f'{prefix}_fixations'] is not None:
                                fixations_by_word, word_measures = trial[f'{prefix}_fixations']
                                for i, word in enumerate(fixations_by_word):
                                    # drop 'FPOGV'
                                    contained_fixations = word['contained_fixations'].drop('FPOGV', axis=1)
                                    if contained_fixations.empty:
                                        print(f'Warning: empty word for trial {participant} {tdf['File']} {tdf["Session.Trial.UID"]}')
                                        # add a single np.nan
                                        contained_fixations = pd.DataFrame(np.nan, index=[0], columns=contained_fixations.columns)

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
                                    contained_fixations['CYCLE'] = tdf['Cycle.ID']
                                    contained_fixations['PARTICIPANT'] = f'P{int(str(tdf['Participant']).split('-')[0]):02d}'

                                    match = re.match(r'^(\d+)\.data\.processed$', tdf['File'])
                                    if match:
                                        session = int(match.group(1))

                                    contained_fixations['SESSION'] = session
                                    contained_fixations['SESSION_DATE'] = tdf['Date']
                                    contained_fixations['SESSION_START_TIME'] = tdf['Time']
                                    contained_fixations['TRIAL_IN_SESSION'] = tdf['Session.Trial.UID']
                                    contained_fixations['WORD'] = word['text']
                                    contained_fixations['CONDITION'] = tdf['Condition']
                                    contained_fixations['RELATION'] = tdf['Relation']
                                    contained_fixations['RELATION_CATEGORY'] = word['relation_letter']
                                    contained_fixations['HAS_DIFFERENTIAL_REINFORCEMENT'] = tdf['HasDifferentialReinforcement']
                                    contained_fixations['RESULT'] = tdf['Result']
                                    contained_fixations['RESPONSE'] = tdf['Response'].strip()
                                    contained_fixations['LATENCY'] = tdf['Latency']

                                    dfs.append(contained_fixations)

        df = pd.concat(dfs, ignore_index=True)
        mask = df[['X', 'Y']].notna().all(axis=1)
        df.loc[mask, 'LETTER_CLUSTER'] = kmeans.fit_predict(df.loc[mask, ['X', 'Y']])
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0])  # Sorts left-to-right
        label_mapping = {original: new_label+1 for new_label, original in enumerate(sorted_indices)}
        df['LETTER_CLUSTER'] = df['LETTER_CLUSTER'].map(label_mapping, na_action='ignore')
        participants.append(df)
        # plot_cluster(df, kmeans)

    exporting = pd.concat(participants, ignore_index=True)
    exporting['RESPONSE'] = exporting['RESPONSE'].str.replace('-4', '')
    # export to csv
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_fixations_CD.csv')
    exporting.to_csv(filename, index=False)










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
    response : str
    if trial['relation'] == 'CD':
        response = trial['response'].strip().replace('-4', '')
        contained_fixations['RESULT'] = word['text'] == response
    else:
        response = trial['response'].strip()
        contained_fixations['RESULT'] = trial['result']
    contained_fixations['RESPONSE'] = response
    contained_fixations['LATENCY'] = trial['latency']
    contained_fixations['SECTION'] = section
    contained_fixations['LETTER_CLUSTER'] = letter_cluster

    return contained_fixations

def export_raw_fixations_all_relations():
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

                    trials = load_pickle(filename+'_raw_fixations')

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
    filename = os.path.join(folder, f'{foldername_2}_raw_fixations_all_relations.csv')
    exporting.to_csv(filename, index=False)



def export_raw_fixations():
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
                                events = trial['data']

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

def export_cluster_stats():
    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_fixations_CD.csv')
    df = pd.read_csv(filename)

    # Calculate cluster statistics using groupby and unstack
    cluster_stats = (
        df.groupby(['PARTICIPANT', 'SESSION', 'TRIAL_IN_SESSION', 'LETTER_CLUSTER'])
        .agg(
            count=('DURATION', 'size'),
            duration=('DURATION', 'sum'))
        .reset_index()
    )

    cluster_stats = cluster_stats.pivot_table(
        index=['PARTICIPANT', 'SESSION', 'TRIAL_IN_SESSION'],
        columns='LETTER_CLUSTER',
        values=['count', 'duration'],
        fill_value=0
    )

    # Flatten multi-level columns
    cluster_stats.columns = [
        f'letter{int(cluster)}_count' if stat == 'count' else f'letter{int(cluster)}_duration'
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

if __name__ == '__main__':
    # export_all_relations()
    # export_CD()
    # export_raw_fixations()
    # plot_clusters()
    export_cluster_stats()