import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from explorer import (
    load_data,
    export
)

df = load_data()

color_map = {
    '1': 'gray',
    '2a': 'gray',
    '2b': 'gray',
    '3': 'black',
    '4': 'black',
    '5': 'black',
    '6': 'black',
    '7': 'black'
}
df['edgecolors'] = df['condition'].map(color_map)

facecolor_map = {
    '1': 'gray',
    '2a': 'gray',
    '2b': 'gray',
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

column_names = [
    'comparisons1_fixations_count',
    'comparisons2_fixations_count',
    'comparisons3_fixations_count',
    'comparisons4_fixations_count']
df['comparisons_fixations_count'] = df[column_names].sum(axis=1)

column_names = [
    'comparisons1_fixations_duration',
    'comparisons2_fixations_duration',
    'comparisons3_fixations_duration',
    'comparisons4_fixations_duration']
df['comparisons_fixations_duration'] = df[column_names].sum(axis=1)

column_names = [
    'comparisons1_letters_switchings',
    'comparisons2_letters_switchings',
    'comparisons3_letters_switchings',
    'comparisons4_letters_switchings']
df['comparisons_letters_switchings'] = df[column_names].sum(axis=1)

lines = []
# normalize session column
i = 0
for participant, participant_data in df.groupby('participant'):
    # exclude conditions 3 to 7
    teaching = participant_data[participant_data['condition'] != '3']
    teaching = teaching[teaching['condition'] != '4']
    teaching = teaching[teaching['condition'] != '5']
    teaching = teaching[teaching['condition'] != '6']
    teaching = teaching[teaching['condition'] != '7']

    ##############################################
    # teaching AB hits
    ##############################################
    teaching_ab = teaching[teaching['relation'] == 'AB']
    teaching_ab_hits = teaching_ab[teaching_ab['result'] == 1]['result'].count()
    total_teaching_ab_trials = teaching_ab['result'].count()

    # teaching AB duration
    teaching_ab_duration = teaching_ab['duration'].sum()

    # teaching AB fixations, duration and switchings
    teaching_ab_fixations_count = teaching_ab['comparisons_fixations_count'].sum()
    teaching_ab_fixations_duration = teaching_ab['comparisons_fixations_duration'].sum()
    teaching_ab_switchings = teaching_ab['comparisons_switchings'].sum()

    ##############################################
    # teaching AC hits
    ##############################################
    teaching_ac = teaching[teaching['relation'] == 'AC']
    teaching_ac_hits = teaching_ac[teaching_ac['result'] == 1]['result'].count()
    total_teaching_ac_trials = teaching_ac['result'].count()

    # teaching AC duration
    teaching_ac_duration = teaching_ac['duration'].sum()

    # teaching AC fixations, duration and switchings
    teaching_ac_fixations_count = teaching_ac['comparisons_fixations_count'].sum()
    teaching_ac_fixations_duration = teaching_ac['comparisons_fixations_duration'].sum()
    teaching_ac_switchings = teaching_ac['comparisons_switchings'].sum()
    teaching_ac_letters_switchings = teaching_ac['comparisons_letters_switchings'].sum()

    ##############################################
    # teaching CD hits
    ##############################################
    teaching_cd = teaching[teaching['relation'] == 'CD']
    teaching_cd_hits = teaching_cd[teaching_cd['result'] == 1]['result'].count()
    total_teaching_cd_trials = teaching_cd['result'].count()

    # teaching CD duration
    teaching_cd_duration = teaching_cd['duration'].sum()

    # teaching CD fixations, duration and switchings
    teaching_cd_fixations_count = teaching_cd['sample1_fixations_count'].sum()
    teaching_cd_fixations_duration = teaching_cd['sample1_fixations_duration'].sum()
    teaching_cd_letters_switchings = teaching_cd['sample1_letters_switchings'].sum()

    ##############################################
    # teaching AC-CD
    ##############################################
    teaching_ac_cd = teaching[teaching['relation'] != 'AB']
    teaching_ac_cd_hits = teaching_ac_cd[teaching_ac_cd['result'] == 1]['result'].count()
    total_teaching_ac_cd_trials = teaching_ac_cd['result'].count()

    # teaching AC-CD duration
    teaching_ac_cd_duration = teaching_ac_cd['duration'].sum()


    ##############################################
    # probes bc, teaching words
    ##############################################
    probes_teaching_words = participant_data[participant_data['condition'] == '3']
    probes_bc_teaching_words = probes_teaching_words[probes_teaching_words['relation'] == 'BC']

    # probes bc, teaching words hits
    probes_bc_teaching_words_hits = probes_bc_teaching_words[probes_bc_teaching_words['result'] == 1]['result'].count()
    probes_bc_teaching_words_total_trials = probes_bc_teaching_words['result'].count()

    # probes bc, teaching words duration
    probes_bc_teaching_words_duration = probes_bc_teaching_words['duration'].sum()

    # probes bc, teaching words fixations, duration and switchings
    probes_bc_teaching_words_fixations_count = probes_bc_teaching_words['comparisons_fixations_count'].sum()
    probes_bc_teaching_words_fixations_duration = probes_bc_teaching_words['comparisons_fixations_duration'].sum()
    probes_bc_teaching_words_switchings = probes_bc_teaching_words['comparisons_switchings'].sum()
    probes_bc_teaching_words_letters_switchings = probes_bc_teaching_words['comparisons_letters_switchings'].sum()

    ##############################################
    # probes cb, teaching words
    ##############################################
    probes_teaching_words = participant_data[participant_data['condition'] == '3']
    probes_cb_teaching_words = probes_teaching_words[probes_teaching_words['relation'] == 'CB']

    # probes cb, teaching words hits
    probes_cb_teaching_words_hits = probes_cb_teaching_words[probes_cb_teaching_words['result'] == 1]['result'].count()
    probes_cb_teaching_words_total_trials = probes_cb_teaching_words['result'].count()

    # probes cb, teaching words duration
    probes_cb_teaching_words_duration = probes_cb_teaching_words['duration'].sum()

    # probes cb, teaching words fixations, duration and switchings
    probes_cb_teaching_words_fixations_count = probes_cb_teaching_words['comparisons_fixations_count'].sum()
    probes_cb_teaching_words_fixations_duration = probes_cb_teaching_words['comparisons_fixations_duration'].sum()
    probes_cb_teaching_words_switchings = probes_cb_teaching_words['comparisons_switchings'].sum()
    probes_cb_teaching_words_letters_switchings = probes_cb_teaching_words['comparisons_letters_switchings'].sum()

    ##############################################
    # probes bc, assessment words
    ##############################################
    probes_assessment_words = participant_data[participant_data['condition'] == '4']
    probes_bc_assessment_words = probes_assessment_words[probes_assessment_words['relation'] == 'BC']

    # probes bc, assessment words hits
    probes_bc_assessment_words_hits = probes_bc_assessment_words[probes_bc_assessment_words['result'] == 1]['result'].count()
    probes_bc_assessment_words_total_trials = probes_bc_assessment_words['result'].count()

    # probes bc, assessment words duration
    probes_bc_assessment_words_duration = probes_bc_assessment_words['duration'].sum()

    # probes bc, assessment words fixations, duration and switchings
    probes_bc_assessment_words_fixations_count = probes_bc_assessment_words['comparisons_fixations_count'].sum()
    probes_bc_assessment_words_fixations_duration = probes_bc_assessment_words['comparisons_fixations_duration'].sum()
    probes_bc_assessment_words_switchings = probes_bc_assessment_words['comparisons_switchings'].sum()
    probes_bc_assessment_words_letters_switchings = probes_bc_assessment_words['comparisons_letters_switchings'].sum()

    ##############################################
    # probes cb, assessment words
    ##############################################
    probes_assessment_words = participant_data[participant_data['condition'] == '4']
    probes_cb_assessment_words = probes_assessment_words[probes_assessment_words['relation'] == 'CB']

    # probes cb, assessment words hits
    probes_cb_assessment_words_hits = probes_cb_assessment_words[probes_cb_assessment_words['result'] == 1]['result'].count()
    probes_cb_assessment_words_total_trials = probes_cb_assessment_words['result'].count()

    # probes cb, assessment words duration
    probes_cb_assessment_words_duration = probes_cb_assessment_words['duration'].sum()

    # probes cb, assessment words fixations, duration and switchings
    probes_cb_assessment_words_fixations_count = probes_cb_assessment_words['comparisons_fixations_count'].sum()
    probes_cb_assessment_words_fixations_duration = probes_cb_assessment_words['comparisons_fixations_duration'].sum()
    probes_cb_assessment_words_switchings = probes_cb_assessment_words['comparisons_switchings'].sum()
    probes_cb_assessment_words_letters_switchings = probes_cb_assessment_words['comparisons_letters_switchings'].sum()

    ##############################################
    # probes CD hits
    ##############################################
    probes_cd = participant_data[participant_data['condition'] == '5']
    probes_cd_hits = probes_cd[probes_cd['result'] == 1]['result'].count()
    probes_cd_total_trials = probes_cd['result'].count()

    # probes CD duration
    probes_cd_duration = probes_cd['duration'].sum()

    # probes CD fixations, duration and switchings
    probes_cd_fixations_count = probes_cd['sample1_fixations_count'].sum()
    probes_cd_fixations_duration = probes_cd['sample1_fixations_duration'].sum()
    probes_cd_letters_switchings = probes_cd['sample1_letters_switchings'].sum()

    ##############################################
    # probes AC hits
    ##############################################
    probes_ac = participant_data[participant_data['condition'] == '6']
    probes_ac_hits = probes_ac[probes_ac['result'] == 1]['result'].count()
    probes_ac_total_trials = probes_ac['result'].count()

    # probes AC duration
    probes_ac_duration = probes_ac['duration'].sum()
    # max_duration = probes_ac['duration'].max()

    # probes AC fixations, duration and switchings
    probes_ac_fixations_count = probes_ac['comparisons_fixations_count'].sum()
    probes_ac_fixations_duration = probes_ac['comparisons_fixations_duration'].sum()
    probes_ac_switchings = probes_ac['comparisons_switchings'].sum()
    probes_ac_letters_switchings = probes_ac['comparisons_letters_switchings'].sum()

    ##############################################
    # multiple probes procedure hits
    ##############################################
    multiple_probes_procedure = participant_data[participant_data['condition'] == '7']
    multiple_probes_procedure_hits = multiple_probes_procedure[multiple_probes_procedure['result'] == 1]['result'].count()
    multiple_probes_procedure_total_trials = multiple_probes_procedure['result'].count()

    # multiple probes procedure duration
    multiple_probes_procedure_duration = multiple_probes_procedure['duration'].sum()

    # multiple probes procedure fixations, duration and switchings
    multiple_probes_procedure_fixations_count = multiple_probes_procedure['sample1_fixations_count'].sum()
    multiple_probes_procedure_fixations_duration = multiple_probes_procedure['sample1_fixations_duration'].sum()
    multiple_probes_procedure_letters_switchings = multiple_probes_procedure['sample1_letters_switchings'].sum()

    # add to list
    lines.append((
        participant,

        multiple_probes_procedure_duration,
        multiple_probes_procedure_hits/multiple_probes_procedure_total_trials,
        multiple_probes_procedure_fixations_duration,
        multiple_probes_procedure_fixations_count,
        multiple_probes_procedure_letters_switchings,

        teaching_ab_duration,
        teaching_ab_hits/total_teaching_ab_trials,
        teaching_ab_fixations_duration,
        teaching_ab_fixations_count,
        teaching_ab_switchings,

        teaching_ac_duration,
        teaching_ac_hits/total_teaching_ac_trials,
        teaching_ac_fixations_duration,
        teaching_ac_fixations_count,
        teaching_ac_switchings,
        teaching_ac_letters_switchings,

        teaching_cd_duration,
        teaching_cd_hits/total_teaching_cd_trials,
        teaching_cd_fixations_duration,
        teaching_cd_fixations_count,
        teaching_cd_letters_switchings,

        teaching_ac_cd_duration,
        teaching_ac_cd_hits/total_teaching_ac_cd_trials,

        probes_bc_teaching_words_duration,
        probes_bc_teaching_words_hits/probes_bc_teaching_words_total_trials,
        probes_bc_teaching_words_fixations_duration,
        probes_bc_teaching_words_fixations_count,
        probes_bc_teaching_words_switchings,
        probes_bc_teaching_words_letters_switchings,

        probes_cb_teaching_words_duration,
        probes_cb_teaching_words_hits/probes_cb_teaching_words_total_trials,
        probes_cb_teaching_words_fixations_duration,
        probes_cb_teaching_words_fixations_count,
        probes_cb_teaching_words_switchings,
        probes_cb_teaching_words_letters_switchings,

        probes_bc_assessment_words_duration,
        probes_bc_assessment_words_hits/probes_bc_assessment_words_total_trials,
        probes_bc_assessment_words_fixations_duration,
        probes_bc_assessment_words_fixations_count,
        probes_bc_assessment_words_switchings,
        probes_bc_assessment_words_letters_switchings,

        probes_cb_assessment_words_duration,
        probes_cb_assessment_words_hits/probes_cb_assessment_words_total_trials,
        probes_cb_assessment_words_fixations_duration,
        probes_cb_assessment_words_fixations_count,
        probes_cb_assessment_words_switchings,
        probes_cb_assessment_words_letters_switchings,

        probes_cd_duration,
        probes_cd_hits/probes_cd_total_trials,
        probes_cd_fixations_duration,
        probes_cd_fixations_count,
        probes_cd_letters_switchings,

        probes_ac_duration,
        probes_ac_hits/probes_ac_total_trials,
        probes_ac_fixations_duration,
        probes_ac_fixations_count,
        probes_ac_switchings,
        probes_ac_letters_switchings,
    ))

# save to file
header = [
    'participant',

    'multiple_probes_procedure_duration',
    'multiple_probes_procedure_hits',
    'multiple_probes_procedure_fixations_duration',
    'multiple_probes_procedure_fixations_count',
    'multiple_probes_procedure_letters_switchings',

    'teaching_ab_duration',
    'teaching_ab_hits',
    'teaching_ab_fixations_duration',
    'teaching_ab_fixations_count',
    'teaching_ab_switchings',

    'teaching_ac_duration',
    'teaching_ac_hits',
    'teaching_ac_fixations_duration',
    'teaching_ac_fixations_count',
    'teaching_ac_switchings',
    'teaching_ac_letters_switchings',

    'teaching_cd_duration',
    'teaching_cd_hits',
    'teaching_cd_fixations_duration',
    'teaching_cd_fixations_count',
    'teaching_cd_letters_switchings',

    'teaching_ac_cd_duration',
    'teaching_ac_cd_hits',

    'probes_bc_teaching_words_duration',
    'probes_bc_teaching_words_hits',
    'probes_bc_teaching_words_fixations_duration',
    'probes_bc_teaching_words_fixations_count',
    'probes_bc_teaching_words_switchings',
    'probes_bc_teaching_words_letters_switchings',

    'probes_cb_teaching_words_duration',
    'probes_cb_teaching_words_hits',
    'probes_cb_teaching_words_fixations_duration',
    'probes_cb_teaching_words_fixations_count',
    'probes_cb_teaching_words_switchings',
    'probes_cb_teaching_words_letters_switchings',

    'probes_bc_assessment_words_duration',
    'probes_bc_assessment_words_hits',
    'probes_bc_assessment_words_fixations_duration',
    'probes_bc_assessment_words_fixations_count',
    'probes_bc_assessment_words_switchings',
    'probes_bc_assessment_words_letters_switchings',

    'probes_cb_assessment_words_duration',
    'probes_cb_assessment_words_hits',
    'probes_cb_assessment_words_fixations_duration',
    'probes_cb_assessment_words_fixations_count',
    'probes_cb_assessment_words_switchings',
    'probes_cb_assessment_words_letters_switchings',

    'probes_cd_duration',
    'probes_cd_hits',
    'probes_cd_fixations_duration',
    'probes_cd_fixations_count',
    'probes_cd_switchings',

    'probes_ac_duration',
    'probes_ac_hits',
    'probes_ac_fixations_duration',
    'probes_ac_fixations_count',
    'probes_ac_switchings',
    'probes_ac_letters_switchings'
]

lines = sorted(lines, key=lambda x: x[2], reverse=True)
# calculate normalized values for duration, count, and switchings along columns
df = pd.DataFrame(lines, columns=header)
for column in df.columns:
    if 'duration' in column or 'count' in column or 'switchings' in column:
        df[column+'_norm'] = df[column]/df[column].max()

export(df, 'ranking.csv')

# drop participant column
columns = []
for column in df.columns:
    if 'ac_cd' in column or 'bc' in column or 'cb' in column:
        columns.append(column)
        continue

    if 'norm' in column:
        continue

    if 'participant' in column or 'duration' in column or 'count' in column or 'switchings' in column:
        columns.append(column)

df = df.drop(columns, axis=1)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Find the strongest pairwise correlation
corr_abs = correlation_matrix.abs()
np.fill_diagonal(corr_abs.values, 0)  # Exclude self-correlation
max_corr = corr_abs.unstack().idxmax()
strongest_pair = (max_corr, correlation_matrix.loc[max_corr])

# Unstack the correlation matrix to create a list of all correlations
correlation_pairs = corr_abs.unstack()

# Sort the correlations from strongest to weakest
sorted_correlations = correlation_pairs.sort_values(ascending=False).drop_duplicates()

# Display the sorted correlations
sorted_correlations.reset_index(name='Correlation').rename(columns={'level_0': 'Variable 1', 'level_1': 'Variable 2'})

i = 0
for index, (var1, var2) in enumerate(sorted_correlations.index):
    value = sorted_correlations.loc[(var1, var2)]
    print(f"The correlation ranked #{index + 1} is between {var1} and {var2} with a value of {value}")
    i += 1
    if i == 9:
        break

# Plot heatmap for visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="viridis")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join('output', 'correlation_heatmap.pdf'))