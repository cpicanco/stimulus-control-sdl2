participants = [
    '1-BAL24',
    '2-SAB24',
    '3-BAP23',
    '4-DIE25',
    '5-TUC23',
    '6-DAD22',
    '7-CHO19',
    '8-PARE23',
    '9-DIZ25',
    '10-NEC25',
    '11-CAB18',
    '12-CAN22',
    '13-PRE24',
    '14-BRI24',
    '15-PUR24',
    '16-MIR25',
    '17-CHI20',
    '18-TIA24',
    '19-JOA23',
    '20-PAT24',
    '21-MAT22',
    '22-RAI25',
    '23-PED25',
    '24-GIO24',
    '25-LUC25',
    '26-LCS25',
    '27-RAI17',
    '28-BEA25',
    '29-FEN21'
]

participants_groups = {
    '1_BAL24': 'A',
    '2_SAB24': 'A',
    '3_BAP23': 'C',
    '4_DIE25': 'B',
    '5_TUC23': 'B',
    '6_DAD22': 'B',
    '7_CHO19': 'B',
    '8_PARE23': 'B',
    '9_DIZ25': 'C',
    '10_NEC25': 'B',
    '11_CAB18': 'C',
    '12_CAN22': 'A',
    '13_PRE24': 'B',
    '14_BRI24': 'B',
    '15_PUR24': 'A',
    '16_MIR25': 'C',
    '17_CHI20': 'B',
    '18_TIA24': 'A',
    '19_JOA23': 'C',
    '20_PAT24': 'C',
    '21_MAT22': 'C',
    '22_RAI25': 'C',
    '23_PED25': 'C',
    '24_GIO24': 'A',
    '25_LUC25': 'A',
    '26_LCS25': 'A',
    '27-RAI17': 'A',
    '28_BEA25': 'A',
    '29_FEN21': 'B'
}

trial_code_of_groups = {
    'C': {'C-B':[49, 50],            'B-C':[47, 48]}, # desconhecidos
    'B': {'C-B':[49, 50, 109, 110],   'B-C':[47, 48]}, #
    'A': {'C-B':[109, 110],          'B-C':[47, 48]} # conhecidas
}

participants_to_ignore = []
participants_exceptions = [
    # there was differential reinforcement for these two participants in the last 30 trials
    '4_DIE25',
    '5_TUC23',
]
foldername = 'estudo4'

other_participants_groups = {
    # '11-BLI24' : 'B',
    '12-CIR21' : 'A',
    # '21-MAR23' : 'B',
    # '22-DIS20' : 'B',
    '23-PAR22' : 'A',
    # '13-POS23' : 'B',
    '14-RAC20' : 'B',
    '19-GEO22' : 'B',
    '20-ANN23' : 'B',
    '4-BON24' : 'C',
    '5-CHA22' : 'C',
    '6-BTC22' : 'A',
    '8-ADO22' : 'C',
    '9-CAX24' : 'C'
}
other_participants_folder = 'estudo3'

if __name__ == '__main__':
    from collections import Counter

    participants_groups = {**participants_groups, **other_participants_groups}
    value_counts = Counter(participants_groups.values())
    print(value_counts)