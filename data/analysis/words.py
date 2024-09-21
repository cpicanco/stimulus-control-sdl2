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

if __name__ == '__main__':
    print(len([word for word in recombine_letters()]))