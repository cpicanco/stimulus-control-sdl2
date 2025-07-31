import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    # plot_histogram(df)


    hits_df = df.pivot_table(
        index='participant',
        columns='group',
        values='result',
        aggfunc='sum'
    )


    hits_per_cycle_df = df.pivot_table(
        index='participant',
        columns='cycle',
        values='result',
        aggfunc='sum'
    )

    count_per_cycle_df = df.pivot_table(
        index='participant',
        columns='cycle',
        values='result',
        aggfunc='count'
    )

    hit_proportion_per_cycle_df = hits_per_cycle_df / count_per_cycle_df


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

    delta_df = pd.concat([mean_df, count_df, hits_df, mean_delta_df], axis=1)
    delta_df['left_right_difference'] = delta_df['left'] - delta_df['right']
    delta_df['novelty_index'] = (delta_df['left'] - delta_df['right']) - ((delta_df['teaching'] - 0.5) * 2)

    delta_df = delta_df.reset_index()
    delta_df = delta_df.sort_values(by='left_right_difference', ascending=False)

    # print mean of delta_df['difference']
    print(f'Index Mean: {delta_df['left_right_difference'].mean()}')
    print(f'Index Min.: {delta_df['left_right_difference'].min()}')
    print(f'Index Max.: {delta_df['left_right_difference'].max()}')

    delta_df['hit_proportion'] = delta_df['total_hits'] / delta_df['total_count']

    print(f'Hit Proportion Mean: {delta_df["hit_proportion"].mean()}')
    print(f'Hit Proportion Min.: {delta_df["hit_proportion"].min()}')
    print(f'Hit Proportion Max.: {delta_df["hit_proportion"].max()}')

    print(f'Teaching Mean: {delta_df["teaching_delta"].mean()}')
    print(f'Teaching Min.: {delta_df["teaching_delta"].min()}')
    print(f'Teaching Max.: {delta_df["teaching_delta"].max()}')

    # list delta_df["hit_proportion"] per participant
    for participant in delta_df["participant"].unique():
        print(f'Participant {participant} hit_proportion: {delta_df[delta_df["participant"] == participant]["hit_proportion"].mean()}')


    y = delta_df['hit_proportion'].to_list()
    x = delta_df['left_right_difference'].to_list()


    plot_correlation(x, y,
                        'Diferença entre esquerda e direita\nnas palavras de generalização',
                        'Proporção de acertos',
                        'left_right_delta',
                        save=True)

    x = delta_df['teaching_delta'].to_list()

    plot_correlation(x, y,
                     'Diferença entre esquerda e direita\nnas palavras de ensino',
                     'Proporção de acertos',
                     'teaching_delta',
                     save=True, limit_x=True)

    x = delta_df['left_delta'].to_list()

    plot_correlation(x, y,
                     'Diferença entre esquerda e direita\nnas palavras de generalização (novo à esquerda)',
                     'Proporção de acertos',
                     'left_delta',
                     save=True, limit_x=True)

    x = delta_df['right_delta'].to_list()

    plot_correlation(x, y,
                     'Diferença entre esquerda e direita\nnas palavras de generalização (novo à direita)',
                     'Proporção de acertos',
                     'right_delta',
                     save=True, limit_x=True)


    # exclude participants with left_count or right_count < 10
    delta_df = delta_df[(delta_df['left_count'] >= 10) & (delta_df['right_count'] >= 10)]

    # Get the sorted participant order from delta_df
    participant_order = delta_df['participant'].tolist()
    difference_order = delta_df['left_right_difference'].tolist()

    # Convert 'participant' to a categorical type with the custom order
    df_sorted = df.copy()
    df_sorted['participant'] = pd.Categorical(
        df_sorted['participant'],
        categories=participant_order,  # Enforce the desired order
        ordered=True
    )

    # Sort the DataFrame by the categorical 'participant' order
    df_sorted = df_sorted.sort_values('participant')

    # save sorted df to csv
    table_path = os.path.join('tables', 'df.csv')
    df_sorted.to_csv(table_path, index=False)




    def block_mean(values, block_len):
        """
        Split `values` into consecutive blocks of length `block_len`
        and return the mean of each block.  Any leftover trials at the
        tail that cannot fill a complete block are ignored.
        """
        values = np.array(values)
        n_blocks = len(values) // block_len
        values = values[:n_blocks * block_len]          # trim tail
        return values.reshape(n_blocks, block_len).mean(axis=1)


    fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    fig.set_size_inches(5.27, 10)
    axes = axes.flatten()

    n = 0
    for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
        print(f"Participant: {participant} Difference: {difference}")

        lines = []
        for group, group_data in cd_data.groupby('group', sort=False):
            if group_data.empty:
                raise ValueError(f"No data found for {participant}, {group}")

            group_data = group_data.sort_values(['session', 'trial_in_session'])

            trials = [i+1 for i in range(len(group_data))]
            index = group_data['index'].values
            print(group_data['word'].unique())


            # # apply a moving average to smooth the data
            # window_size = 4
            # moving_average = np.convolve(index, np.ones(window_size)/window_size, mode='valid')
            # # Calculate how many NaNs to pad (for odd window_size)
            # n_pad = window_size - 1

            # # Pad with NaNs at the beginning
            # index = np.concatenate([
            #     np.full(n_pad, np.nan),
            #     moving_average])

            if group == 'left':
                window = 4
                index = index.reshape(-1, window).mean(axis=1).repeat(window)[:len(index)]
                line, = axes[n].plot(trials, index,
                            linewidth=1,
                            linestyle='-',
                            color='k',
                            label='Sílaba nova na esquerda')
                lines.append(line)

            elif group == 'right':
                window = 4
                index = index.reshape(-1, window).mean(axis=1).repeat(window)[:len(index)]
                line, = axes[n].plot(trials, index,
                            linewidth=1,
                            linestyle='--',
                            color='k',
                            label='Sílaba nova na direita')
                lines.append(line)
            elif group == 'teaching':
                # array size
                window = 12
                if index.size % window != 0:
                    index = np.concatenate([index, np.full((-index.size) % window , np.nan)])
                    trials = np.array(trials)
                    trials = np.concatenate([trials, np.full((-trials.size) % window, np.nan)])

                index = index.reshape(-1, window).mean(axis=1).repeat(window)[:len(index)]
                line, = axes[n].plot(trials, index,
                            linewidth=1,
                            linestyle='-',
                            color='gray',
                            label='Palavra ensinada')
                lines.append(line)



        # write participant number
        axes[n].text(1.0, 1.0, participant, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

        axes[n].set_ylim(0, 1)
        axes[n].set_yticks([0, 0.5, 1])

        axes[n].set_xlim(1, len(trials))
        axes[n].set_xticks(range(4, len(trials)+1, 4))

        axes[n].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)

        axes[n].spines['top'].set_visible(False)
        axes[n].spines['right'].set_visible(False)
        axes[n].spines['bottom'].set_visible(False)
        axes[n].spines['left'].set_visible(False)

        n += 1
    # exclude empty axes from fig
    # axes = [ax for ax in axes if not ax.lines]
    # for ax in axes:
    #     fig.delaxes(ax)

    # Add x and y labels in the middle of figure
    fig.text(0.5, 0.02, 'Tentativas', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Proporção de fixações na sílaba da esquerda', ha='center', va='center', fontsize=12, rotation=90)

    handles = [lines[0], lines[1]]
    labels = [line.get_label() for line in handles]
    plt.tight_layout()
    fig.legend(handles, labels,
            loc='upper left',
            ncol=4,
            columnspacing=1.5,
            handletextpad=0.5)

    plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # Show the plot
    extension = 'svg'
    filename = f'isr_all_participants_pt_br.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()









    fig, axes = plt.subplots(11, 2, sharex=True, sharey=True)
    fig.set_size_inches(5.27, 10)
    axes = axes.flatten()

    n = 0
    for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
        print(f"Participant: {participant} Difference: {difference}")

        cd_data = cd_data.sort_values(['session', 'trial_in_session'])
        cd_data['uid'] = range(1, len(cd_data) + 1)
        cd_data['uid_group'] = (cd_data['uid'] - 1)  // 4
        cd_data['target_calculation'] = cd_data['result'].groupby(cd_data['uid_group']).transform('mean') * 100

        lines = []
        for group, group_data in cd_data.groupby('group', sort=False):
            if group_data.empty:
                raise ValueError(f"No data found for {participant}, {group}")

            group_data = group_data.sort_values(['uid'])
            index = group_data['target_calculation'].tolist()

            if group == 'left':
                line, = axes[n].plot(group_data['uid'], index,
                            linewidth=1,
                            linestyle='-',
                            color='k',
                            label='Sílaba nova na esquerda')
                lines.append(line)

            elif group == 'right':
                line, = axes[n].plot(group_data['uid'], index,
                            linewidth=1,
                            linestyle='--',
                            color='gray',
                            label='Sílaba nova na direita')
                lines.append(line)

            elif group == 'teaching':
                line, = axes[n].plot(group_data['uid'], index,
                            linewidth=1,
                            linestyle=':',
                            color='gray',
                            label='Palavras de ensino')
                lines.append(line)


        # write participant number
        axes[n].text(1.0, 1.0, participant, ha='right', va='bottom', transform=axes[n].transAxes, fontsize=10)

        # axes[n].set_ylim(0, 1)
        # axes[n].set_yticks([0, 0.5, 1])

        axes[n].set_xlim(1, len(cd_data['uid']))
        axes[n].set_xticks(range(20, len(cd_data['uid'])+1, 20))

        # axes[n].axhline(y=50, color='gray', linestyle='--', linewidth=0.5)

        axes[n].spines['top'].set_visible(False)
        axes[n].spines['right'].set_visible(False)
        axes[n].spines['bottom'].set_visible(False)
        axes[n].spines['left'].set_visible(False)

        n += 1


    # Add x and y labels in the middle of figure
    fig.text(0.5, 0.02, 'Tentativas', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Leitura oral (Porcentagem de acertos)', ha='center', va='center', fontsize=12, rotation=90)

    handles = [lines[2], lines[0], lines[1]]
    labels = [line.get_label() for line in handles]
    plt.tight_layout()
    fig.legend(handles, labels,
            loc='upper left',
            ncol=3,
            columnspacing=1.5,
            handletextpad=0.5)

    plt.subplots_adjust(top=0.95, left=0.12, bottom=0.06, hspace=0.8)
    # Show the plot
    extension = 'svg'
    filename = f'isr_all_participants_percent_left_right_teaching_pt_br.{extension}'
    print(filename)
    plt.savefig(filename, format=extension, dpi=300, transparent=False)
    plt.close()








    # data_dir()
    # cd(os.path.join('analysis', 'output', 'fixations_over_letters_hits'))


    # # normalize session column
    # for (participant, cd_data), difference in zip(df_sorted.groupby('participant', sort=False), difference_order):
    #     print(f"Participant: {participant} Difference: {difference}")

    #     cd_data = cd_data.sort_values(['session', 'trial_in_session'])
    #     cd_data['uid'] = range(1, len(cd_data) + 1)
    #     # responses = cd_data['response']
    #     fig, axes = plt.subplots(3, 1, figsize=(7, 4), sharex=False, gridspec_kw={'height_ratios': [1, 4, 1]})

    #     ###############################
    #     ########## TOP PLOT ###########
    #     ###############################

    #     tdf = cd_data[cd_data['group'] == 'left']
    #     words = tdf['reference_word']
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

    #     index_line, = axes[1].plot(x, tdf['index'],
    #                 linewidth=1,
    #                 color='k',
    #                 label='New syllabels on the left')
    #     axes[1].grid(axis='x')
    #     lines.append(index_line)




    #     tdf = cd_data[cd_data['group'] == 'right']
    #     words = tdf['reference_word']

    #     index_line, = axes[1].plot(x, tdf['index'],
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

    #     extension = 'svg'
    #     filename = f'isr_{participant}_fixations_detailed.{extension}'
    #     print(filename)
    #     plt.savefig(filename, format=extension, dpi=300, transparent=False)
    #     plt.close()
