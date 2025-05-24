# from collections import deque

def nwise_non_overlapping(iterable, n):
    it = iter(iterable)
    return list(zip(*[it] * n))

# def create_words(syllables, custom_ordering=None, swap_syllables=False, number_of_syllables=2):
#     words = nwise_non_overlapping(syllables, number_of_syllables)
#     ordering = custom_ordering
#     if ordering is None:
#         ordering = [tuple(list(range(len(word)))) for word in words]

#     if swap_syllables:
#         ordering = [tuple(reversed(order)) for order in ordering]

#     words = [tuple([word[i] for i in order]) for order, word in zip(ordering, words)]
#     return words

# def rotate(syllables, n=1):
#     # loop through it pairwise
#     d = deque(syllables)
#     d.rotate(n)
#     return list(d)

# def custom_protocol(number_of_syllables=4, number_of_syllables_per_word=2, number_of_words_per_cycle=2):
#     syllables1 = [(i, i) for i in range(number_of_syllables)]
#     syllables2 = rotate(syllables1, -1)
#     syllables3 = rotate(syllables2[:number_of_syllables-1], 1) + [syllables2[-1]]

#     base_protocol = [
#         syllables1,
#         syllables2,
#         syllables3
#     ]

#     first_half = [create_words(
#         syllables=syllables,
#         swap_syllables=False,
#         number_of_syllables=number_of_syllables_per_word) for syllables in base_protocol]

#     second_half = [create_words(
#         syllables=syllables,
#         swap_syllables=True,
#         number_of_syllables=number_of_syllables_per_word) for syllables in base_protocol]

#     protocol = first_half + second_half
#     flat_protocol = [word for cycle in protocol for word in cycle]
#     protocol = [(flat_protocol[i:i+number_of_words_per_cycle]) for i in range(0, len(flat_protocol), number_of_words_per_cycle)]

#     return protocol


# def general_protocol(
#         number_of_syllables=4,
#         number_of_syllables_per_word=2,
#         number_of_words_per_cycle=2,
#         allow_equal_syllables=False):

#     base = [(i, i) for i in range(number_of_syllables)]
#     base_protocol = []
#     for syllable in base:
#         for syllable2 in base:
#             if allow_equal_syllables:
#                 base_protocol.append((syllable, syllable2))
#             else:
#                 if syllable != syllable2:
#                     base_protocol.append((syllable, syllable2))


#     for _ in range(number_of_syllables):
#         base_protocol.append(rotate(base_protocol[-1], -1))


#     first_half = [create_words(
#         syllables=syllables,
#         swap_syllables=False,
#         number_of_syllables=number_of_syllables_per_word) for syllables in base_protocol]

#     second_half = [create_words(
#         syllables=syllables,
#         swap_syllables=True,
#         number_of_syllables=number_of_syllables_per_word) for syllables in base_protocol]

#     protocol = first_half + second_half
#     flat_protocol = [word for cycle in protocol for word in cycle]
#     protocol = [(flat_protocol[i:i+number_of_words_per_cycle]) for i in range(0, len(flat_protocol), number_of_words_per_cycle)]

#     return protocol

# def teaching_protocol(
#         number_of_syllables=4,
#         number_of_syllables_per_word=2,
#         number_of_words_per_cycle=2,
#         protocol_name='Picanco_et_al_2025'):

#     if protocol_name == 'Hanna_et_al_2011':
#         number_of_syllables = 4
#         number_of_syllables_per_word = 2
#         number_of_words_per_cycle = 2

#         protocol = custom_protocol(
#             number_of_syllables=number_of_syllables,
#             number_of_syllables_per_word=number_of_syllables_per_word,
#             number_of_words_per_cycle=number_of_words_per_cycle)

#     elif protocol_name == 'Picanco_et_al_2025':
#         protocol = general_protocol(
#             number_of_syllables=number_of_syllables,
#             number_of_syllables_per_word=number_of_syllables_per_word,
#             number_of_words_per_cycle=number_of_words_per_cycle)

#     else:
#         if protocol_name == 'diagonal':
#             possible_words = number_of_syllables**number_of_syllables_per_word
#             possible_words_no_diagonal = possible_words - number_of_syllables
#             possible_words_without_repetition = possible_words // number_of_words_per_cycle

#         elif protocol_name == 'diagonal_with_overlap':
#             pass

#         elif protocol_name == 'diagonal_without_repetition':
#             pass


#     return protocol

# def new_protocol(syllables=['BO', 'FA', 'LE', 'NI', 'TU']):
#     consonants = [syllable[0] for syllable in syllables]
#     vowels = [syllable[1] for syllable in syllables]

#     # Define the pattern for generating each row's words
#     protocol = teaching_protocol(
#         number_of_syllables=len(syllables),
#         number_of_syllables_per_word=2,
#         number_of_words_per_cycle=2)
#     table = []
#     for row_idx, cycle_pattern in enumerate(protocol):
#         words = []
#         for word in cycle_pattern:
#             text = ''
#             for (c, v) in word:
#                 text += consonants[c]+vowels[v]
#             words.append(text)
#         table.append((row_idx+1, [word for word in words]))
#     return table

# def syllab_to_ipa_text(syllab):
#     return syllab['c']['ipa'] + syllab['v']['ipa']

# def syllab_to_hr_text(syllab, strees=False):
#     if strees:
#         return syllab['c']['hr'] + syllab['v']['hrs']
#     else:
#         return syllab['c']['hr'] + syllab['v']['hr']

# def syllabs_to_ipa_text(syllab1, syllab2):
#     return syllab_to_ipa_text(syllab1) + '.ˈ' + syllab_to_ipa_text(syllab2)

# def syllabs_to_hr_text(syllab1, syllab2, strees=False):
#     return syllab_to_hr_text(syllab1, strees) + syllab_to_hr_text(syllab2, strees)

# def syllabs_are_different(syllab1, syllab2):
#     return syllab1['c']['hr'] != syllab2['c']['hr'] or syllab1['v']['hr'] != syllab2['v']['hr']

# def goldstein_matrix(unit):
#     print('\t'+'\t'.join(unit))
#     for s1 in unit:
#         print(s1, end='\t')
#         for s2 in unit:
#             print(syllabs_to_ipa_text(s1, s2), end='\t')
#         print()

# def goldstein_matrix(syllabs, IPA=True):
#     if IPA:
#         print('\t'+'\t'.join([syllab_to_ipa_text(s) for s in syllabs]))
#     else:
#         print('\t'+'\t'.join([syllab_to_hr_text(s) for s in syllabs]))
#     for s1 in syllabs:
#         if IPA:
#             print(syllab_to_ipa_text(s1), end='\t')
#         else:
#             print(syllab_to_hr_text(s1), end='\t')
#         for s2 in syllabs:
#             if IPA:
#                 print(syllabs_to_ipa_text(s1, s2), end='\t')
#             else:
#                 print(syllabs_to_hr_text(s1, s2), end='\t')
#         print()

# def save_to_tab_delimited_file(syllabs, filename, IPA=False):
#     with open(filename, 'w', encoding='utf-8') as f:
#         if IPA:
#             f.write('\t'+'\t'.join([syllab_to_ipa_text(s) for s in syllabs]) + '\n')
#         else:
#             f.write('\t'+'\t'.join([syllab_to_hr_text(s) for s in syllabs]) + '\n')
#         for s1 in syllabs:
#             if IPA:
#                 f.write(syllab_to_ipa_text(s1) + '\t')
#             else:
#                 f.write(syllab_to_hr_text(s1) + '\t')
#             for s2 in syllabs:
#                 if IPA:
#                     f.write(syllabs_to_ipa_text(s1, s2) + '\t')
#                 else:
#                     f.write(syllabs_to_hr_text(s1, s2) + '\t')
#             f.write('\n')

# # consonants
# PlosiveBilabial = {'ipa':'b', 'hr':'b'}
# NonSibilantFricative = {'ipa':'f', 'hr':'f'}
# LateralApproximantAlveolar = {'ipa':'l', 'hr':'l'}
# NasalAlveolar = {'ipa':'n', 'hr':'n'}

# # vowels
# OpenFront = {'ipa':'a', 'hr':'a', 'hrs':'á'}
# OpenMidFront = {'ipa':'ɛ', 'hr':'e', 'hrs':'é'}
# CloseFront = {'ipa':'i', 'hr':'i', 'hrs':'í'}
# OpenMidBack = {'ipa':'ɔ', 'hr':'o', 'hrs':'ó'}

# Consonants = [PlosiveBilabial, NonSibilantFricative, LateralApproximantAlveolar, NasalAlveolar]
# Vowels = [OpenFront, OpenMidFront, CloseFront, OpenMidBack]
# Syllabs = [{'c':c, 'v':v} for c in Consonants for v in Vowels]

# # goldstein_matrix(Syllabs)

# Words = [{'ipa': syllabs_to_ipa_text(s1, s2), 'hrs':syllabs_to_hr_text(s1, s2, True), 'hr':syllabs_to_hr_text(s1, s2, False)} \
#         for s1 in Syllabs for s2 in Syllabs if syllabs_are_different(s1, s2)]


# if __name__ == '__main__':
#     save_to_tab_delimited_file(Syllabs, 'syllabs.txt')



# import random

# def generate_syllable_word_pairs():
#     """
#     Generates pairs of two-syllable words using a specific set of
#     consonant-vowel syllables, ensuring each syllable is used exactly once
#     across all words before pairing.
#     """

#     # 1. Initialization
#     consonants = ['b', 'f', 'l', 'n']
#     vowels = ['a', 'e', 'i', 'o']

#     # 2. Generate All Syllables
#     syllables = []
#     for c in consonants:
#         for v in vowels:
#             syllables.append(c + v)
#     # print(f"Generated Syllables ({len(syllables)}): {syllables}") # Optional debug print

#     # 3. Prepare for Pairing (Ensure Uniqueness)
#     if len(syllables) % 2 != 0:
#         raise ValueError("The number of syllables must be even to form pairs.")

#     shuffled_syllables = syllables[:] # Create a copy to shuffle
#     # random.shuffle(shuffled_syllables)
#     # # print(f"Shuffled Syllables: {shuffled_syllables}") # Optional debug print

#     # 4. Form Words
#     words = []
#     # Iterate through the shuffled list, taking 2 syllables at a time
#     for i in range(0, len(shuffled_syllables), 2):
#         syllable1 = shuffled_syllables[i]
#         syllable2 = shuffled_syllables[i+1]

#         if syllable1 == syllable2:
#              raise ValueError(f"Cannot form word with repeated syllable: {syllable1}")
#         word = syllable1 + syllable2
#         words.append(word)
#     # print(f"Generated Words ({len(words)}): {words}") # Optional debug print

#     # 5. Form Word Pairs
#     if len(words) % 2 != 0:
#         raise ValueError("The number of words must be even to form pairs.")

#     word_pairs = []
#     # Iterate through the word list, taking 2 words at a time
#     for i in range(0, len(words), 2):
#         word1 = words[i]
#         word2 = words[i+1]
#         word_pairs.append((word1, word2)) # Store pair as a tuple

#     # 6. Output (Return the pairs)
#     return word_pairs

# # --- Execution ---
# if __name__ == "__main__":
#     generated_pairs = generate_syllable_word_pairs()

#     print("Algorithm Output: Word Pairs")
#     print("-----------------------------")
#     if not generated_pairs:
#         print("No pairs were generated.")
#     else:
#         for i, pair in enumerate(generated_pairs):
#             print(f"Pair {i+1}: ({pair[0]}, {pair[1]})")

import itertools

def permutations(mask):
    """
    Generate (i, j) pairs of all possible permutations of a given mask,
    preserving the order, forward first, backward second.

    For example, using the diagonal of a 4x4 matrix as input mask:
    [(0, 0), (1, 1), (2, 2), (3, 3)]

    It returns the following (i, j) pairs:
    [(0, 0), (1, 1), (0, 0), (2, 2), (0, 0), (3, 3), (1, 1), (2, 2), (1, 1), (3, 3), (2, 2), (3, 3),
     (1, 1), (0, 0), (2, 2), (0, 0), (3, 3), (0, 0), (2, 2), (1, 1), (3, 3), (1, 1), (3, 3), (2, 2)]

    We assume that the mask is a list of tuples (row, col) representing the indices of the matrix.

    And the following matrix:
    ['ba', 'be', 'bi', 'bo']
    ['fa', 'fe', 'fi', 'fo']
    ['la', 'le', 'li', 'lo']
    ['na', 'ne', 'ni', 'no']

    The output can then be used to return the corresponding cells:
    [('ba', 'fe'), ('ba', 'li'),
     ('ba', 'no'), ('fe', 'li'),
     ('fe', 'no'), ('li', 'no'),
     ('fe', 'ba'), ('li', 'ba'),
     ('no', 'ba'), ('li', 'fe'),
     ('no', 'fe'), ('no', 'li')]

    """
    ordered_matrix_pairs = []
    # creates forward and mirrored index pairs for iterating the mask
    forward_pairs = list(itertools.combinations(range(len(mask)), 2))
    mirrored_pairs = [(next, item) for (item, next) in forward_pairs]
    def process_section(pairs, is_mirrored):
        chunks = [pairs[i:i+2] for i in range(0, len(pairs), 2)]
        for idx, chunk in enumerate(chunks):
            for item, next in chunk:
                ordered_matrix_pairs.append((mask[item][0], mask[item][1]))
                ordered_matrix_pairs.append((mask[next][0], mask[next][1]))

            # skip the last mirrored pair
            if is_mirrored and idx == len(chunks) - 1:
                continue

    process_section(forward_pairs, is_mirrored=False)
    process_section(mirrored_pairs, is_mirrored=True)

    return ordered_matrix_pairs

def is_isogram(letters):
    return len(letters) == len(set(letters))

def find_all_isograms(l, n, sort=False):
    if len(l) % n != 0:
        print(f"List length must be divisible by {n}")
        return

    if n > 1:
        chunk_size = len(l) // n
        lists = [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]
    else:
        lists = [l, l]

    solution = [c for c in itertools.product(*lists) if is_isogram(''.join(c))]

    if sort:
        solution.sort()

    return solution

def word_from_grid(word, grid):
    return ''.join([grid[c[0]][c[1]] for c in word])

# assing constants
vowels = ('a', 'e', 'i', 'o')
consonants = ('b', 'f', 'l', 'n')

# create a 4x4 grid vowels in columns and consonants in rows
grid = []
for c in consonants:
    row = [c+v for v in vowels]
    grid.append(row)

[print(grid[row]) for row in range(len(grid))]

# Example usage with the default mask
default_mask = [(0, 0), (1, 1), (2, 2), (3, 3)]

user_mask = [(3, 2), (2, 1), (0, 3), (1, 0)]
ordered_cells = permutations(user_mask)
words = []
for cells in nwise_non_overlapping(ordered_cells, 2):
    words.append(word_from_grid(cells, grid))

[print(word) for word in words]

print(find_all_isograms(words, 2))


# print(find_all_isograms(words, 2))