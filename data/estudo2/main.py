import os
import shutil

import pandas as pd

def cd(directory, verbose=True):
    os.chdir(directory)
    if verbose:
        print("Current Working Directory: ", os.getcwd())

transcript_file = os.path.join('output', 'estudo2','probes_CD.data.processed')
df = pd.read_csv(transcript_file, sep='\t', encoding='utf-8', engine='python')
df['Session'] = df['File'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
df.drop(columns=[
    'Speech',
    'Speech-2',
    'Speech-3',
    'Report.Timestamp',
    'Session.Block.UID',
    'Session.Block.Trial.UID',
    'Session.Block.ID',
    'Session.Block.Trial.ID',
    'Session.Block.Name',
    'Cycle.ID',
    'Relation',
    'Comparisons',
    'Result',
    'CounterHit',
    'CounterHit.MaxConsecutives',
    'CounterMiss',
    'CounterMiss.MaxConsecutives',
    'CounterNone',
    'CounterNone.MaxConsecutives',
    'Sample-Position.1',
    'Comparison-Position.1',
    'Comparison-Position.2',
    'Comparison-Position.3',
    'Response',
    'HasDifferentialReinforcement',
    'Trial.ID',
    'Latency',
    'Condition',
    'Date',
    'Time',
    'File'
    ], inplace=True)

df['Speech'] = ''
# change columns order
df = df[['ID', 'Participant', 'Name',  'Session', 'Session.Trial.UID', 'Speech']]

script_dir = os.path.dirname(os.path.abspath(__file__))
cd(script_dir)

# ensure deterministic results for randomness
seeds = {}
for participant, participant_df in df.groupby('Participant'):
    cd(script_dir, verbose=False)
    wav_output_dir = os.path.join(script_dir,'output', participant)
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir)
    name = participant.replace('\\', '').replace('-', '_')
    xlsx_file = os.path.join(script_dir,'output', f'{name}.xlsx')
    participant_number = int(participant.split('-')[0])
    cd(os.path.join(script_dir, participant, 'responses'), verbose=False)

    seeds[name] = -1
    seed = 1
    while True:
        valid = True
        files_to_copy = []
        sampled_df = participant_df.sample(frac=0.3, random_state=seed) # select 10% of rows at random for each participant

        sampled_df.sort_values(by=['Session', 'Session.Trial.UID'], inplace=True)

        for _, row in sampled_df.iterrows():
            session = int(row['Session'])
            trial = int(row['Session.Trial.UID'])
            word = row['Name']
            filename = f'Speech-P{participant_number:02d}-S{session:02d}-B01-T{trial:02d}-C01-R01-{word}.wav'
            if not os.path.isfile(filename):
                print(f'File {filename} does not exist.')
                valid = False
                break

            files_to_copy.append(filename)

        if not valid:
            seed += 1
            print(participant, seed,  '*****************************************')
            continue

        if len(sampled_df) != len(files_to_copy):
            raise Exception(f'Number of files {len(files_to_copy)} does not match number of rows {len(sampled_df)}')

        sampled_df.to_excel(xlsx_file, index=False)
        for filename in files_to_copy:
            # copy file, create folder if it doesn't exist
            shutil.copy2(filename, os.path.join(wav_output_dir, filename))

        seeds[name] = seed
        break

    # save seeds to file
    seeds_file = os.path.join(script_dir,'output', 'seed_per_participant_dict.txt')
    with open(seeds_file, 'w', encoding='utf-8') as f:
        f.write(str(seeds))