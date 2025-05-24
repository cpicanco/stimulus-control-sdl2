participants = [
    '11-BLI24', # Engineering
    '12-CIR21', # Engineering
    '13-POS23', # Engineering
    '14-RAC20', # Engineering
    '15-le23',  # Engineering
    '19-GEO22', # Engineering
    '20-ANN23', # Engineering
    '4-BON24',  # Engineering
    '5-CHA22',  # Engineering
    '6-BTC22',  # Engineering
    '8-ADO22',  # Engineering
    '9-CAX24',  # Engineering
    '23-PAR22', # Engineering
    '21-MAR23', # Engineering
    '22-DIS20'  # Engineering
]

participants_groups = {
    '4_BON24': 'C',
    '5_CHA22': 'C',
    '6_BTC22': 'C',
    '8_ADO22': 'C',
    '9_CAX24': 'C',
    '11_BLI24': 'A',
    '12_CIR21': 'A',
    '13_POS23': 'B',
    '14_RAC20': 'B',
    '15_le23': 'B',
    '19_GEO22': 'B',
    '20_ANN23': 'B',
    '23_PAR22': 'A',
    '21_MAR23': 'A',
    '22_DIS20': 'A'
}

trial_code_of_groups = {
    'C': {'C-B':[49, 50],            'B-C':[47, 48]}, # desconhecidos
    'B': {'C-B':[49, 50, 109, 110],   'B-C':[47, 48]}, #
    'A': {'C-B':[109, 110],          'B-C':[47, 48]} # conhecidas
}

participants_to_ignore = ['15-le23']

foldername = 'estudo3'