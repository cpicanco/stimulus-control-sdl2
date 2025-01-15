from enum import Enum

import pandas as pd
import numpy as np
# from scipy.stats import entropy
# from itertools import groupby

from player_information import GazeInfo
from player_behavior_events import (BehavioralEvents, TrialEvents)
from player_gaze_events import GazeEvents
from player_constants import DEFAULT_FPS

from player_drawings import (
    Screen,
    Rectangle
)

from player_utils import (
    as_dict
)

# create a data filter for shapes
#    mask = shape.contains_points(gaze_data)
#    data = all_gaze_data[mask]

def calculate_mesh(rectangle, n_vertical_split=1, number_of_letters=4):
    horizontal_bins = np.linspace(0, rectangle.width, num=number_of_letters+1)

    vertical_bins = np.linspace(0, rectangle.height, num=n_vertical_split+1)

    return np.meshgrid(horizontal_bins, vertical_bins)


def mixed_strings_to_bool(input_array):
    """
        Convert a list of strings to a boolean array
        where True indicates a change in the string.
        A change is counted when the change is to a non-empty and different string.
    """
    output_array = np.zeros(len(input_array), dtype=bool)
    # Find indices of non-empty strings
    non_empty_indices = np.where(input_array != '')[0]
    if len(non_empty_indices) == 0:
        return output_array

    non_empty_values = input_array[non_empty_indices]
    # Initialize the changes array
    changes = np.zeros(len(non_empty_values), dtype=bool)
    changes[0] = False  # The first item is always False
    # Compare each non-empty string with the previous non-empty string
    changes[1:] = non_empty_values[1:] != non_empty_values[:-1]
    # Assign the results back to the output array at the correct positions
    output_array[non_empty_indices] = changes
    return output_array

def count_trues_in_groups(input_list, group_size=150):
    # Convert the input list to a NumPy array for efficient processing
    bool_array = np.array(input_list, dtype=bool)
    # Calculate the number of groups needed
    num_groups = int(np.ceil(len(bool_array) / group_size))
    # Pad the array if necessary to make it divisible by group_size
    padding_size = num_groups * group_size - len(bool_array)
    if padding_size > 0:
        bool_array = np.pad(bool_array, (0, padding_size), mode='constant', constant_values=False)
    # Reshape the array into groups
    grouped_array = bool_array.reshape(num_groups, group_size)
    # Count the number of True values in each group
    true_counts = np.sum(grouped_array, axis=1)
    # Convert the result to a list
    return true_counts.tolist()

def mix_vertically_str(array):
    result = np.empty(array.shape[1], dtype=object)  # Initialize result array
    result[:] = ''  # Fill with empty strings initially

    for i in range(array.shape[1]):
        column = array[:, i]
        non_empty = column[column != '']  # Select non-empty strings
        if len(np.unique(non_empty)) > 1:
            # Raise an error if there's more than one unique non-empty string
            raise ValueError(f"Conflict detected at position {i}: {non_empty}")
        if len(non_empty) > 0:
            result[i] = non_empty[0]  # Assign the unique non-empty string

    return result

class WordFilterType(Enum):
    ByPositive = 1
    ByNegative = 2
    ByPositiveAndNegative = 3

class TrialFilterType(Enum):
    TrialsReferenceWord = 1
    TrialsAll = 2

class TimeFilter:
    def __init__(self, data, unit):
        self._data = data
        self._unit = unit
        self._mask = None
        self._current = 0
        self.duration = self._data['timestamp'][-1]

    def _update_mask(self, start: float, duration: float):
        self.mask = (self._data['timestamp'] >= start) & (self._data['timestamp'] < duration)

    @property
    def data(self):
        return self._data[self.mask]

    @property
    def range(self):
        return np.arange(0, self.duration, self._unit)

    @property
    def current(self: int) -> int:
        return self._current

    @current.setter
    def current(self, value: int):
        self._current = value
        self._update_mask(value, value + self._unit)

class WordFilter:
    def __init__(self,
                 info: GazeInfo,
                 screen: Screen,
                 time_sorted_data: np.ndarray,
                 gaze: GazeEvents,
                 behavior: BehavioralEvents,
                 trials: TrialEvents,
                 word_string: str = '',
                 ):
        self._info = info
        self._screen = screen
        self._mask = None
        self._data = time_sorted_data
        self._gaze = gaze
        self._behavior = behavior
        self._trials = trials

        self.word_string = word_string
        self.stimuli_begin = None
        self.stimuli_end = None
        self.annotation_column = self._behavior.__Annotation__
        self.event_column = self._behavior.__Event__
        self.events = None

        gaze_df = self._gaze.DataFrame
        self.gaze_df = gaze_df[gaze_df[self._gaze.__FixationValid__]]
        self.data_by_trial = None
        self.events_pattern = None
        self.stimuli_begin = None
        self.stimuli_end = None
        self.setup_target_trial_events()

    def setup_target_trial_events(self,
                                  target_relations: list = None,
                                  target_stimuli_types: list = None):

        """
        :param target_relations:
            AB, AC, BC, CB, CD, BB

        :param target_stimuli_types:
            Samp, Comp
        :return:
        """
        events = self._info.target_trial_events()
        name = self._info.SessionName
        name = name.split('-')[2:]
        name = '-'.join(name)

        self.events = events[name]

        if target_relations is None:
            target_relations = self.events.keys()

        self._session_name = name
        self.target_relations = target_relations
        self.target_stimuli_types = target_stimuli_types

        target_events = []
        for relation in self.events.keys():
            if relation in self.target_relations:
                for stimuli_type in self.events[relation].keys():
                    if target_stimuli_types is None:
                        target_stimuli_types = self.events[relation].keys()

                    self.target_stimuli_types
                    if stimuli_type in target_stimuli_types:
                        event = self.events[relation][stimuli_type]
                        if event not in target_events:
                            target_events.append(event)

        begin_events = [event['begin'] for event in target_events]
        end_events = [event['end'] for event in target_events]
        self.stimuli_begin = self.get_event_pattern(begin_events)
        self.stimuli_end = self.get_event_pattern(end_events)
        self.events_pattern = self.get_event_pattern(begin_events+end_events)

    def get_event_pattern(self, events: list = ['']):
        return r'('+'|'.join(list(set(events)))+')'

    @property
    def events_grouped_by_trial(self):
        dataframe = self._behavior.DataFrame
        dataframe = dataframe[~dataframe[self.event_column].str.match('LastStimuli.Show', na=False)]
        return dataframe.groupby('Session.Trial.UID')

    @property
    def switchings_per_second(self) -> np.ndarray:
        return np.concatenate([trial['switchings_per_second'] for trial in self.data_by_trial])

    @property
    def switching_rate(self) -> list:
        return [trial['switching_rate'] for trial in self.data_by_trial]

    def trials(self, calculate_fixations: bool = False) -> list:
        def calculate_heatmap(data: dict):
            relation_letter = data['relation_letter']
            rectangle = data['rectangle']
            x_data = data['relative_fixations']['FPOGX'].values
            y_data = data['relative_fixations']['FPOGY'].values
            weights = data['relative_fixations']['FPOGD'].values

            width = rectangle.width
            height = rectangle.height

            n_vertical_split = 1
            if relation_letter == 'A':
                n_horizontal_slit = 1
            elif relation_letter == 'B':
                n_horizontal_slit = 1
            elif relation_letter == 'C':
                n_horizontal_slit = 4
            else:
                raise Exception(f"Relation letter not supported: {relation_letter}")

            horizontal_bins = np.linspace(0, width, num=n_horizontal_slit+1)

            vertical_bins = np.linspace(0, height, num=n_vertical_split+1)

            # invert y
            # y_data = height - y_data

            heatmap_data, _, _ = np.histogram2d(
                x_data, y_data,
                bins=[horizontal_bins, vertical_bins],
                range=[[0, width], [0, height]],
                weights=weights)

            X, Y = np.meshgrid(horizontal_bins, vertical_bins)

            data['heatmap'] = heatmap_data
            data['heatmap_xmesh'] = X
            data['heatmap_ymesh'] = Y
            data['heatmap_relation_letter'] = relation_letter

        # def clamp(gaze: pd.DataFrame) -> pd.DataFrame:
        #     gaze = gaze[gaze['FPOGX'] >= 0.0]
        #     gaze = gaze[gaze['FPOGY'] >= 0.0]
        #     gaze = gaze[gaze['FPOGX'] <= 1.0]
        #     gaze = gaze[gaze['FPOGY'] <= 1.0]
        #     return gaze

        def process_fixations(gaze: pd.DataFrame) -> pd.DataFrame:
            denormalized_fixations = denormalize(gaze['FPOGX'], gaze['FPOGY'])
            denormalized_fixations = pd.DataFrame({
                'FPOGX': denormalized_fixations[0],
                'FPOGY': denormalized_fixations[1],
                'TIME_TICK': gaze['TIME_TICK'],
                'FPOGD': gaze['FPOGD'],
                'FPOGID': gaze['FPOGID'],
                'FPOGV': gaze['FPOGV']
            })

            processed_rows = []
            timestamps = []
            id = 1
            for FPOGID, fixation in denormalized_fixations.groupby('FPOGID'):
                # skip invalid fixations
                fx = fixation[fixation['FPOGV'] == 1]
                if fx.empty:
                    continue

                # skip fixations too short
                fixation_duration = fx['FPOGD'].max()
                if fixation_duration < 0.060:
                    continue

                # calculate mean position
                processed_rows.append({
                    'ID' : id,
                    'FPOGID': FPOGID,
                    'FPOGD': fixation_duration,
                    'FPOGX': fx['FPOGX'].mean(),
                    'FPOGY': fx['FPOGY'].mean(),
                    'FPOGV': fx['FPOGV'].iloc[0],
                    'TIME_TICK': fx['TIME_TICK'].iloc[0],
                })
                timestamps.append({'ID': id, 'TIME_TICKS': fx['TIME_TICK']})
                id += 1

            return pd.DataFrame(processed_rows), timestamps

        def process_switchings(
            fixations_over_elements: list,
            processed_fixations: pd.DataFrame,
            fixations_timestamps: list
        ) -> dict:
            # list of words "nibo", "fale", "fale", "nibo" ... or letters "n", "i", "b", "o", ...
            elements_text = [element['text'] for element in fixations_over_elements]
            # corresponding contained mask of each word
            contained_masks = [element['contained_mask'] for element in fixations_over_elements]

            # switchings
            contained_per_element = []
            for element, mask in zip(elements_text, contained_masks):
                labels = np.full(mask.shape, '', dtype=object)
                labels[mask] = element
                contained_per_element.append(labels)

            switchings_labels = mix_vertically_str(np.array(contained_per_element))
            switchings_mask = mixed_strings_to_bool(switchings_labels)
            switchings = np.sum(switchings_mask)

            contained_masks = np.any(np.array(contained_masks), axis=0)
            switchings_timestamps = processed_fixations[switchings_mask]['ID']
            switchings_timestamps = [f['TIME_TICKS'] for f in fixations_timestamps if f['ID'] in switchings_timestamps]

            return {
                'contained_masks': contained_masks,
                'switchings_mask': switchings_mask,
                'switchings': switchings,
                'switchings_timestamps': switchings_timestamps,
                'switchings_labels': switchings_labels
            }


        def process_fixations_over_letters(
                letters : list,
                processed_fixations : pd.DataFrame,
                fixations_timestamps : list,
                word_rectangle : Rectangle
            ) -> list:

            letters_rectangles = word_rectangle.split_horizontally()
            fixations_over_letters = []
            for letter, letter_rectangle in zip(letters, letters_rectangles):
                letter_contains = np.vectorize(letter_rectangle.contains)
                letter_fixations_contained_mask = letter_contains(
                    processed_fixations['FPOGX'], processed_fixations['FPOGY'])

                contained = processed_fixations[letter_fixations_contained_mask]
                contained_fixations = pd.DataFrame({
                    'ID': contained['ID'],
                    'FPOGX': contained['FPOGX'],
                    'FPOGY': contained['FPOGY'],
                    'TIME_TICK': contained['TIME_TICK'],
                    'FPOGD': contained['FPOGD'],
                    'FPOGID': contained['FPOGID'],
                    'FPOGV': contained['FPOGV'],
                })

                formated_letter = {
                    'text': letter, # we assume that different letters
                    'rectangle': word_rectangle,
                    'contained_fixations': contained_fixations,
                    'fixations_count': len(contained_fixations),
                    'contained_mask': letter_fixations_contained_mask
                }
                fixations_over_letters.append(formated_letter)

            switchings = process_switchings(
                fixations_over_elements=fixations_over_letters,
                processed_fixations=processed_fixations,
                fixations_timestamps=fixations_timestamps,
            )

            return formated_letter, switchings

        def process_fixations_over_words(
                fixations : tuple,
                reference_word : str,
                word_shapes_dict : dict,
                relation_letter : str
            ) -> list:

            processed_fixations, fixations_timestamps = fixations

            if processed_fixations.empty:
                return []
            fixations_over_words = []
            not_contained_mask = None
            fixations_contained_mask = None
            for word in word_shapes_dict.keys():
                if not_contained_mask is None:
                    not_contained_mask = np.logical_not(fixations_contained_mask)
                else:
                    logical_not = np.logical_not(fixations_contained_mask)
                    not_contained_mask = np.logical_and(not_contained_mask, logical_not)

                # here we are filtering by word in the trial
                if word == reference_word:
                    word_function = 'positive'
                else:
                    word_function = 'negative'

                word_rectangle = Rectangle.from_dict(word_shapes_dict[word])
                contains = np.vectorize(word_rectangle.contains)
                to_word_space = np.vectorize(word_rectangle.to_relative_coordenates)

                fixations_contained_mask = contains(
                    processed_fixations['FPOGX'], processed_fixations['FPOGY'])
                fixations_relative_to_word = to_word_space(
                    processed_fixations['FPOGX'], processed_fixations['FPOGY'])

                contained_fixations = pd.DataFrame({
                    'ID': processed_fixations['ID'][fixations_contained_mask],
                    'FPOGX': processed_fixations['FPOGX'][fixations_contained_mask],
                    'FPOGY': processed_fixations['FPOGY'][fixations_contained_mask],
                    'TIME_TICK': processed_fixations['TIME_TICK'][fixations_contained_mask],
                    'FPOGD': processed_fixations['FPOGD'][fixations_contained_mask],
                    'FPOGID': processed_fixations['FPOGID'][fixations_contained_mask],
                    'FPOGV': processed_fixations['FPOGV'][fixations_contained_mask],
                })
                fixations_count = len(contained_fixations)

                relative_fixations = pd.DataFrame({
                    'ID': processed_fixations['ID'][fixations_contained_mask],
                    'FPOGX': fixations_relative_to_word[0][fixations_contained_mask],
                    'FPOGY': fixations_relative_to_word[1][fixations_contained_mask],
                    'TIME_TICK': processed_fixations['TIME_TICK'][fixations_contained_mask],
                    'FPOGD': processed_fixations['FPOGD'][fixations_contained_mask],
                    'FPOGID': processed_fixations['FPOGID'][fixations_contained_mask],
                    'FPOGV': processed_fixations['FPOGV'][fixations_contained_mask],
                })

                formated_word = {
                    'relation_letter': relation_letter,
                    'text': word,
                    'function': word_function,
                    'rectangle': word_rectangle,
                    'relative_fixations': relative_fixations,
                    'contained_fixations': contained_fixations,
                    'fixations_count': fixations_count,
                    'contained_mask': fixations_contained_mask
                }
                calculate_heatmap(formated_word)

                if relation_letter == 'C':
                   formated_word['letter_fixations'] = process_fixations_over_letters(
                        letters=list(word),
                        processed_fixations=processed_fixations,
                        fixations_timestamps=fixations_timestamps,
                        word_rectangle=word_rectangle
                    )
                fixations_over_words.append(formated_word)

            word_switchings = process_switchings(
                fixations_over_words,
                processed_fixations,
                fixations_timestamps)

            # print('Derived trial measures: ', derived_trial_measures)

            return fixations_over_words, word_switchings

        def process_section(
            fixations : tuple,
            trial_events : pd.DataFrame,
            event_pattern : str,
            section : str,
            relation : str,
            reference_word : str
        ) -> dict:

            mask = trial_events[self.event_column].str.match(event_pattern)
            events = trial_events[mask]

            if len(events) == 0:
                return {}

            timestamps = events[self._behavior.__Timestamp__]
            section_duration = timestamps.max() - timestamps.min()
            if section_duration == 0:
                return {}

            result = {}
            result['duration'] = section_duration

            if not calculate_fixations:
                return result

            processed_fixations, fixations_timestamps = fixations

            mask = processed_fixations[self._gaze.__Timestamp__].between(timestamps.min(), timestamps.max())
            section_fixations = processed_fixations[mask]
            fixations_timestamps = [f for f in fixations_timestamps if f['ID'] in section_fixations['ID']]

            section_fixations = (section_fixations, fixations_timestamps)

            if section == 'trial':
                result['fixations'] = section_fixations
            elif section == 'sample':
                mk = trial_events[self.event_column].str.match(self.events[relation]['samp']['begin'])
                word_shapes_dict = as_dict(trial_events[mk][self.annotation_column].iloc[0])

                result['fixations'] = process_fixations_over_words(
                    fixations=section_fixations,
                    reference_word=reference_word,
                    word_shapes_dict=word_shapes_dict,
                    relation_letter=relation[0]
                    )

            elif section == 'comparisons':
                mk = trial_events[self.event_column].str.match(self.events[relation]['comp']['begin'])
                word_shapes_dict = as_dict(trial_events[mk][self.annotation_column].iloc[0])

                if (relation == 'BC') or (relation == 'CB'):
                    # BC, CB uses simultaneous/0-delayed mts
                    # show sample -> response to sample -> show comparisons (sample stays)
                    # hence, we need to include sample in shapes dict
                    mk = trial_events[self.event_column].str.match(self.events[relation]['samp']['begin'])
                    candidate_dict = as_dict(trial_events[mk][self.annotation_column].iloc[0])
                    sample_shapes_dict = {}
                    for key in candidate_dict.keys():
                        sample_shapes_dict[key+'_'] = candidate_dict[key]
                    word_shapes_dict.update(sample_shapes_dict)

                else:
                    # AB, AC uses successive/0-delayed mts
                    # show sample -> response to sample -> remove sample -> show comparisons
                    # hence, we don't need to include sample in shapes dict
                    pass

                result['fixations'] = process_fixations_over_words(
                    fixations=section_fixations,
                    reference_word=reference_word,
                    word_shapes_dict=word_shapes_dict,
                    relation_letter=relation[1]
                    )
            return result

        # assign events pattern of trial
        if self.events_pattern is None:
            self.setup_target_trial_events()

        if calculate_fixations:
            denormalize = np.vectorize(self._screen.denormalize)
            processed_fixations = process_fixations(self.gaze_df)
        else:
            processed_fixations = None

        trials = []
        for trial_uid, trial_events in self.events_grouped_by_trial:
            trial = self._trials.from_uid(trial_uid)
            relation = trial['Relation']
            reference_word = trial['Name']

            # assign events pattern of of sample
            sample_event_pattern = self.get_event_pattern(
                [self.events[relation]['samp']['begin'],
                 self.events[relation]['samp']['end']]
            )

            # assign events pattern of comparisons
            if relation == 'CD':
                comparison_event_pattern = None
            else:
                comparison_event_pattern = self.get_event_pattern(
                    [self.events[relation]['comp']['begin'],
                    self.events[relation]['comp']['end']]
                )

            sections = ['trial', 'sample', 'comparisons']
            events = [
                self.events_pattern,
                sample_event_pattern,
                comparison_event_pattern]

            data_by_section = {}
            for section, event_pattern in zip(sections, events):
                if event_pattern is None:
                    data_by_section[section] = {'duration': None, 'fixations': None}
                    continue

                data_by_section[section] = process_section(
                    fixations=processed_fixations,
                    trial_events=trial_events,
                    event_pattern=event_pattern,
                    section=section,
                    relation=relation,
                    reference_word=reference_word
                )

            trial_dict = {
                'duration': data_by_section['trial']['duration'],
                'fixations': data_by_section['trial']['fixations'],
                'sample_duration': data_by_section['sample']['duration'],
                'sample_fixations': data_by_section['sample']['fixations'],
                'comparisons_duration': data_by_section['comparisons']['duration'],
                'comparisons_fixations': data_by_section['comparisons']['fixations'],
                'trial_uid_in_session': trial_uid,
                'cycle': self._info.Cycle,
                'relation': relation,
                'reference_word': reference_word,
                'condition': trial['Condition'],
                'result': trial['Result'],
                'file': trial['File'],
                'has_differential_reinforcement': trial['HasDifferentialReinforcement'],
                'participant': trial['Participant'],
                'latency': trial['Latency']
            }

            trials.append(trial_dict)

        return trials


    #         trials.append({
    #             'uid': trial_data['Session.Trial.UID'].iloc[0],
    #             'words': words,
    #             'gaze': gaze,
    #             'reference_word': reference_word,
    #             'fixations_outside_words_mask': not_contained_mask,
    #             'switchings_per_second': switchings_per_second,
    #             'switchings': switchings,
    #             'switchings_mask': switchings_mask,
    #             'switchings_labels': switchings_labels,
    #             'switching_rate': switchings/trial_duration,
    #             'fixations_count': fixations_count,
    #             'fixation_rate': fixations_count/trial_duration,
    #             'duration': trial_duration
    #         })

    #     max_rate = np.max([trial['switching_rate'] for trial in trials])
    #     max_fixations = np.max([trial['fixation_rate'] for trial in trials])

    #     for trial in trials:
    #         trial['switching_rate_normalized'] = (trial['switching_rate']/max_rate)
    #         trial['switching_max_rate'] = max_rate
    #         trial['fixation_rate_normalized'] = (trial['fixation_rate']/max_fixations)
    #         trial['fixation_max_rate'] = max_fixations

    #     if filter_type == TrialFilterType.TrialsAll:
    #         return trials
    #     elif filter_type == TrialFilterType.TrialsReferenceWord:
    #         return [trial for trial in trials if trial['reference_word'] == self.word_string]


    @property
    def duration_per_trial(self) -> list:
        return [trial['duration'] for trial in self.data_by_trial]

    @property
    def switching_rate_normalized_per_trial(self) -> list:
        return [trial['switching_rate_normalized'] for trial in self.data_by_trial]

    def process_all_trials(self):
        self.data_by_trial = self._filter_by_trial(TrialFilterType.TrialsAll)

    def process_trials_with_reference_word(self):
        self.data_by_trial = self._filter_by_trial(TrialFilterType.TrialsReferenceWord)

    def get_heatmap_of_positive_words(self):
        return self.get_heatmap(WordFilterType.ByPositive)

    def get_heatmap_of_negative_words(self):
        return self.get_heatmap(WordFilterType.ByNegative)

    def get_heatmap_by_trial(self):
        return self.get_heatmap(WordFilterType.ByPositiveAndNegative)

    def get_heatmap(self, filter_type: WordFilterType):
        def larger(rectangles):
            return max(rectangles, key=lambda rectangle: rectangle.width*rectangle.height)

        result = {}
        result['heatmap_by_trial'] = []
        result['heatmaps_by_trial'] = []
        result['rectangles_by_trial'] = []
        for trial in self.data_by_trial:
            data_to_include = {'words': []}
            for word in trial['words']:
                if filter_type == WordFilterType.ByPositive:
                    if word['function'] == 'positive':
                        data_to_include['words'].append(word)
                elif filter_type == WordFilterType.ByNegative:
                    if word['function'] == 'negative':
                        data_to_include['words'].append(word)
                elif filter_type == WordFilterType.ByPositiveAndNegative:
                    if word['function'] == 'positive' or word['function'] == 'negative':
                        data_to_include['words'].append(word)

            if data_to_include['words'] == []:
                # print(f'No words for trial {trial["uid"]} with filter {filter_type}. Creating an empty heatmap.')
                data_to_include['trial_heatmaps'] = [np.zeros((4, n_vertical_split))]
                data_to_include['trials_rectangles'] = [Rectangle(width=514, height=230)]
            else:
                data_to_include['trial_heatmaps'] = [word['heatmap'] for word in data_to_include['words']]
                data_to_include['trials_rectangles'] = [word['rectangle'] for word in data_to_include['words']]
            result['heatmap_by_trial'].append(sum(data_to_include['trial_heatmaps']))
            result['heatmaps_by_trial'].append(data_to_include['trial_heatmaps'])
            result['rectangles_by_trial'].append(larger(data_to_include['trials_rectangles']))

        result['rectangle'] = larger(result['rectangles_by_trial'])
        X, Y = calculate_mesh(result['rectangle'])
        result['heatmap_xmesh'] = X
        result['heatmap_ymesh'] = Y
        result['heatmap'] = sum(result['heatmap_by_trial'])
        return result

class Synchronizer:
    def __init__(self, code: str):
        self.info = GazeInfo(code)
        self.screen = Screen(self.info.ScreenWidth, self.info.ScreenHeight)

        self.behavior = BehavioralEvents(self.info)
        self.behavior.label = 0
        self.behavior_indices = self.behavior.indices

        self.gaze = GazeEvents(self.info)
        self.gaze.label = 1
        self.gaze_indices = self.gaze.indices

        self.trials = TrialEvents(self.info)
        # self.trials.label = 2
        # self.trials_indices = self.trials.indices

        self.desired_fps: int = DEFAULT_FPS

        dtype = [
            ('timestamp', self.gaze.timestamps.dtype),
            ('label', self.gaze.labels.dtype),
            ('index', self.gaze.indices.dtype)]

        data1 = np.array(list(zip(
            self.gaze.timestamps,
            self.gaze.labels,
            self.gaze.indices)), dtype=dtype)
        data2 = np.array(list(zip(
            self.behavior.timestamps,
            self.behavior.labels,
            self.behavior.indices)), dtype=dtype)

        self._data = np.concatenate((data1, data2))
        self._data.sort(order='timestamp')
        self.per_second = TimeFilter(self._data, 1.0)
        self.per_frame = TimeFilter(self._data, 1/self.desired_fps)
        self.duration = self._get_duration()

    def _get_duration(self):
        return np.max((self.gaze.duration(), self.behavior.duration()))

    def word_filter(self, word: str = '') -> WordFilter:
        return WordFilter(
            info=self.info,
            screen=self.screen,
            time_sorted_data=self._data,
            gaze=self.gaze,
            behavior=self.behavior,
            trials=self.trials,
            word_string=word)

    def duration_difference(self) -> float:
        return self.gaze.duration() - self.behavior.duration()

    def frame(self, value):
        self.per_frame.current = value
        return self.per_frame.data

    @property
    def frames(self):
        for value in self.per_frame.range:
            self.per_frame.current = value
            yield value, self.per_frame.data

    def second(self, value):
        self.per_second.current = value
        return self.per_second.data

    @property
    def seconds(self):
        for value in self.per_second.range:
            self.per_second.current = value
            yield value, self.per_second.data

if __name__ == '__main__':
    import os

    from explorer import (
        load_participant_sources,
        load_pickle,
        save_pickle
    )

    sources = load_participant_sources()
    for participant in sources:
        target_sources = []
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                # if 'Pre-treino' not in source['name']:
                if 'Sondas-CD' in source['name']:
                    filename = '_'.join([participant.replace('-', '_'), source['code']])
                    print(filename)
                    # check if file exists
                    if os.path.exists(filename):
                        source['trials'] = load_pickle(filename)
                    else:
                        # force the creating of directories
                        session : Synchronizer
                        session = source['session']
                        session_filtered = session.word_filter('')
                        source['trials'] = session_filtered.trials(calculate_fixations=True)

                        save_pickle(filename, source['trials'])

                    for trial in source['trials']:
                        if trial['relation'] == 'CD':
                            fixations_by_word, measures = trial['sample_fixations']
                            for word in fixations_by_word:
                                print(word['fixations_count'], word['letter_fixations'][1]['switchings'])