# pre teaching words
bolo = 'bolo'
bala = 'bala'

# teaching words
nibo = 'nibo'
fale = 'fale'
bofa = 'bofa'
leni = 'leni'
lebo = 'lebo'
fani = 'fani'
boni = 'boni'
lefa = 'lefa'
fabo = 'fabo'
nile = 'nile'
bole = 'bole'
nifa = 'nifa'

# CD2 generalization words

nibe = 'nibe'
lofi = 'lofi'
bofi = 'bofi'
nale = 'nale'
leba = 'leba'
nofa = 'nofa'
bona = 'bona'
lefi = 'lefi'
fabe = 'fabe'
nilo = 'nilo'
febi = 'febi'
lano = 'lano'

# CD1 generalization words
lani = 'lani'
febo = 'febo'
nole = 'nole'
bifa = 'bifa'

# constant generalization words
falo = 'falo'
bena = 'bena'

constant = [
    falo,
    bena
]

words_per_cycle = {
    1 : {'teaching': [nibo, fale], 'generalization': [nibe, lofi]},
    2 : {'teaching': [bofa, leni], 'generalization': [bofi, nale]},
    3 : {'teaching': [lebo, fani], 'generalization': [leba, nofa]},
    4 : {'teaching': [boni, lefa], 'generalization': [bona, lefi]},
    5 : {'teaching': [fabo, nile], 'generalization': [fabe, nilo]},
    6 : {'teaching': [bole, nifa], 'generalization': [febi, lano]},
}

category_per_word = {
    nibo : 'Teaching',
    fale : 'Teaching',
    bofa : 'Teaching',
    leni : 'Teaching',
    lebo : 'Teaching',
    fani : 'Teaching',
    boni : 'Teaching',
    lefa : 'Teaching',
    fabo : 'Teaching',
    nile : 'Teaching',
    bole : 'Teaching',
    nifa : 'Teaching',

    nibe : 'Generalization1',
    lofi : 'Generalization1',
    bofi : 'Generalization1',
    nale : 'Generalization1',
    leba : 'Generalization1',
    nofa : 'Generalization1',
    bona : 'Generalization1',
    lefi : 'Generalization1',
    fabe : 'Generalization1',
    nilo : 'Generalization1',
    febi : 'Generalization1',
    lano : 'Generalization1',
    lani : 'Generalization2',
    febo : 'Generalization2',
    nole : 'Generalization2',
    bifa : 'Generalization2',

    falo : 'Generalization3',
    bena : 'Generalization3',
}

cycle_per_word = {
    nibo : 1,
    fale : 1,
    bofa : 2,
    leni : 2,
    lebo : 3,
    fani : 3,
    boni : 4,
    lefa : 4,
    fabo : 5,
    nile : 5,
    bole : 6,
    nifa : 6,
    nibe : 1,
    lofi : 1,
    bofi : 2,
    nale : 2,
    leba : 3,
    nofa : 3,
    bona : 4,
    lefi : 4,
    fabe : 5,
    nilo : 5,
    febi : 6,
    lano : 6,
    lani : None,
    febo : None,
    nole : None,
    bifa : None,
    falo : 7,
    bena : 7,
}

pre_test_12_8 = [
    nibo,
    fale,
    bofa,
    leni,
    lebo,
    fani,
    boni,
    lefa,
    fabo,
    nile,
    bole,
    nifa,

    lani,
    febo,
    nole,
    bifa,
    bona,
    lefi,
    fabe,
    nilo
]

pre_test_hardcoded_order = [
    bena,
    falo,

    nibo,
    fale,
    bofa,
    leni,
    lebo,
    fani,
    boni,
    lefa,
    fabo,
    nile,
    bole,
    nifa,

    nibe,
    lofi,
    bofi,
    nale,
    leba,
    nofa,
    bona,
    lefi,
    fabe,
    nilo,
    febi,
    lano,

    lani,
    febo,
    nole,
    bifa
    ]

all = [
    'Fim.'
]

words_per_file = {
    'Ciclo0-0-Pre-treino': ['X1', 'X2', 'bala', 'bolo'],
    'Ciclo1-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo1-1-Treino-AB': words_per_cycle[1]['teaching'] + constant,
    'Ciclo1-2a-Treino-AC-CD': words_per_cycle[1]['teaching'],
    'Ciclo1-2b-Treino-AC-Ref-Intermitente': words_per_cycle[1]['teaching'],
    'Ciclo1-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[1]['teaching'],
    'Ciclo1-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo1-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[1]['generalization'] + constant,
    'Ciclo1-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[1]['generalization'] + constant,

    'Ciclo2-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo2-1-Treino-AB': words_per_cycle[2]['teaching'] + constant,
    'Ciclo2-2a-Treino-AC-CD': words_per_cycle[2]['teaching'],
    'Ciclo2-2b-Treino-AC-Ref-Intermitente': words_per_cycle[2]['teaching'],
    'Ciclo2-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[2]['teaching'],
    'Ciclo2-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo2-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[2]['generalization'] + constant,
    'Ciclo2-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[2]['generalization'] + constant,

    'Ciclo3-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo3-1-Treino-AB': words_per_cycle[3]['teaching'] + constant,
    'Ciclo3-2a-Treino-AC-CD': words_per_cycle[3]['teaching'],
    'Ciclo3-2b-Treino-AC-Ref-Intermitente': words_per_cycle[3]['teaching'],
    'Ciclo3-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[3]['teaching'],
    'Ciclo3-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo3-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[3]['generalization'] + constant,
    'Ciclo3-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[3]['generalization'] + constant,

    'Ciclo4-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo4-1-Treino-AB': words_per_cycle[4]['teaching'] + constant,
    'Ciclo4-2a-Treino-AC-CD': words_per_cycle[4]['teaching'],
    'Ciclo4-2b-Treino-AC-Ref-Intermitente': words_per_cycle[4]['teaching'],
    'Ciclo4-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[4]['teaching'],
    'Ciclo4-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo4-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[4]['generalization'] + constant,
    'Ciclo4-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[4]['generalization'] + constant,

    'Ciclo5-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo5-1-Treino-AB': words_per_cycle[5]['teaching'] + constant,
    'Ciclo5-2a-Treino-AC-CD': words_per_cycle[5]['teaching'],
    'Ciclo5-2b-Treino-AC-Ref-Intermitente': words_per_cycle[5]['teaching'],
    'Ciclo5-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[5]['teaching'],
    'Ciclo5-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo5-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[5]['generalization'] + constant,
    'Ciclo5-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[5]['generalization'] + constant,

    'Ciclo6-0-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo6-1-Treino-AB': words_per_cycle[6]['teaching'] + constant,
    'Ciclo6-2a-Treino-AC-CD': words_per_cycle[6]['teaching'],
    'Ciclo6-2b-Treino-AC-Ref-Intermitente': words_per_cycle[6]['teaching'],
    'Ciclo6-3-Sondas-BC-CB-Palavras-de-ensino': words_per_cycle[6]['teaching'],
    'Ciclo6-4-Sondas-BC-CB-Palavras-reservadas': constant,
    'Ciclo6-5-Sondas-CD-Palavras-generalizacao-reservadas': words_per_cycle[6]['generalization'] + constant,
    'Ciclo6-6-Sondas-AC-Palavras-generalizacao-reservadas': words_per_cycle[6]['generalization'] + constant,

    'Ciclo6-7-Sondas-CD-Palavras-12-ensino-8-generalizacao': pre_test_12_8,
    'Ciclo6-7-Sondas-CD-Palavras-30-Todas' : pre_test_hardcoded_order,
    'Ciclo2-0-Sondas-CD-Palavras-30-Todas' : pre_test_hardcoded_order
}

def recombine_letters(consonants='bfln', vowels='aeio'):
    sylables = []
    for c in consonants:
        for v in vowels:
            sylables.append(c + v)

    for syllable in sylables:
        for syllable2 in sylables:
            yield syllable + syllable2

for word in recombine_letters():
    all.append(word)


teaching = rf'({nibo}|{fale}|{boni}|{lefa}|{nile}|{bole}|{fani}|{lebo}|{bofa}|{leni}|{nifa}|{fabo})'
assessment = rf'({nale}|{lani}|{febo}|{nole}|{bifa}|{nofa}|{lofi}|{fabe}|{febi}|{lano}|{nibe}|{bofi}|{leba}|{bona}|{lefi}|{nilo})'

isr_left = rf'({nofa}|{nale}|{lani}|{febo}|{nole}|{bifa})'
isr_both = rf'({lofi}|{febi}|{lano}|{bena})'
isr_right = rf'({nibe}|{bofi}|{leba}|{bona}|{lefi}|{nilo}|{falo}|{fabe})'

list_right = [nibe, bofi, leba, bona, lefi, nilo, falo, fabe]
list_left = [nofa, nale, lani, febo, nole, bifa]
list_both = [lofi, febi, lano, bena]
list_teaching = [nibo, fale, bofa, leni, lebo, fani, boni, lefa, fabo, nile, bole, nifa]

if __name__ == '__main__':
    # Create word list
    words = list_teaching

    # Calculate the move cycle for each word
    move_cycles = [(i // 2) + 2 for i in range(12)]

    # Generate header
    header = ['Word'] + [f'Cycle{i}' for i in range(1, 8)]

    # Build table rows
    rows = []
    for i in range(12):
        row = [words[i]]
        for cycle in range(1, 8):
            category = 'Category1' if cycle >= move_cycles[i] else 'Category2'
            row.append(category)
        rows.append(row)

    # Write to TSV file
    with open('word_categories.tsv', 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')

    print("Table successfully written to 'word_categories.tsv'")