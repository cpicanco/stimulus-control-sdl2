import os

import numpy as np
import pandas as pd

from explorer import (
    cache_folder,
    foldername_2,
)

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


def plot(title, df):
    hits_df = df.pivot_table(
        index='participant',
        columns='group',
        values='result',
        aggfunc='sum'
    )

    mean_df = df.pivot_table(
        index='participant',
        columns='group',
        values='index',
        aggfunc='mean'
    )

    mean_delta_df = df.pivot_table(
        index='participant',
        columns='group',
        values='index_delta',
        aggfunc='mean'
    )

    mean_delta_df = mean_delta_df.rename(columns={''
        'left': 'left_delta',
        'right': 'right_delta',
        'teaching': 'teaching_delta'})

    count_df = df.pivot_table(
        index='participant',
        columns='group',
        values='index',
        aggfunc='count'
    )

    hits_df = hits_df.rename(columns={''
        'left': 'left_hits',
        'right': 'right_hits',
        'teaching': 'teaching_hits'})
    hits_df['total_hits'] = (hits_df['left_hits'] + hits_df['right_hits'] + hits_df['teaching_hits'])

    count_df = count_df.rename(columns={
        'left': 'left_count',
        'right': 'right_count',
        'teaching': 'teaching_count'})
    count_df['total_count'] = (count_df['left_count'] + count_df['right_count'] + count_df['teaching_count'])

    delta_df = pd.concat([mean_df, count_df, hits_df], axis=1)
    delta_df['left_right_difference'] = delta_df['left'] - delta_df['right']
    delta_df['novelty_index'] = (delta_df['left'] - delta_df['right']) - ((delta_df['teaching'] - 0.5) * 2)
    delta_df['hit_proportion'] = delta_df['total_hits'] / delta_df['total_count']


    y = delta_df['hit_proportion'].to_list()
    x = delta_df['left_right_difference'].to_list()

    # save x, y to a csv file
    filename = os.path.join('tables', 'noelty_index.csv')
    df = pd.DataFrame({'novelty_index': x, 'hit_proportion': y})
    df.to_csv(filename, index=False)

    print(title, '*********************************************')

    plot_correlation(x, y,
                        'Índice de acompanhamento da sílaba nova\n (diferença média entre novo à esquerda e à direita)',
                        'Índice de desempenho\n(proporção de acertos)',
                        'left_right_delta_cycle_'+str(title),
                        save=True)

if __name__ == '__main__':

    from correlation import plot_correlation
    from words import (
        isr_left,
        isr_both,
        isr_right,
        teaching
    )

    data_dir()
    cd(os.path.join('analysis', 'output', 'fixations_over_letters_v3'))

    folder = cache_folder()
    filename = os.path.join(folder, f'{foldername_2}_processed_fixations.csv')

    df = pd.read_csv(filename)

    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)
    df.loc[:, 'index'] = df.apply(calculate_index, axis=1, use_delta=False)
    df.loc[:, 'index_delta'] = df.apply(calculate_index, axis=1, use_delta=True)
    df.loc[:, 'session'] = (df.groupby('participant')['session']
                    .rank(method='dense')
                    .astype(int))
    max_sessions = df.groupby('participant')['session'].transform('max')
    df.loc[df['session'] == max_sessions, 'cycle'] = 7

    word_filters = [
        df['word'].str.match(isr_left),
        df['word'].str.match(isr_right),
        df['word'].str.match(isr_both),
        df['word'].str.match(teaching)]

    groups = [
        'left',
        'right',
        'both',
        'teaching'
    ]

    df['group'] = np.select(word_filters, groups, default='other')

    # show which words are in which group
    # print(df.groupby('group')['word'].unique())

    df = df[df['condition'] == '7']
    df['result'] = np.where(df['response'] == df['word'], 1, 0)

    cycles_df_filters = [[1, 2, 3, 4, 5, 6, 7]]
    cycles_list = ['_'.join([str(cycle) for cycle in cycles]) for cycles in cycles_df_filters]

    for title, cycle_filter in zip(cycles_list, cycles_df_filters):
        df = df[df['cycle'].isin(cycle_filter)]
        plot(title, df)

    # for cycle, cycle_df in df.groupby('cycle'):
    #     plot(cycle, cycle_df)