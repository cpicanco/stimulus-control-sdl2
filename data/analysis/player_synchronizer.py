import pandas as pd
import numpy as np

from player_information import GazeInfo
from player_behavior_events import BehavioralEvents, TrialEvents
from player_gaze_events import GazeEvents
from player_constants import DEFAULT_FPS

from player_drawings import (
    Screen,
    Rectangle
)

from player_utils import (
    as_dict
)


def mixed_strings_to_bool(input_array):
    """
    Converts a list of strings to a boolean array
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
    # Initialize result array
    result = np.empty(array.shape[1], dtype=object)
    # Fill with empty strings initially
    result[:] = ''
    # loop over columns
    for i in range(array.shape[1]):
        # Get column
        column = array[:, i]
        # Select non-empty strings
        non_empty = column[column != '']
        if len(np.unique(non_empty)) > 1:
            # Raise an error if there's more than one unique non-empty string
            raise ValueError(f"Conflict detected at position {i}: {non_empty}")
        if len(non_empty) > 0:
            # Assign the unique non-empty string
            result[i] = non_empty[0]

    return result

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

class TrialFilter:
    def __init__(self,
                 info: GazeInfo,
                 screen: Screen,
                 time_sorted_data: np.ndarray,
                 gaze: GazeEvents,
                 behavior: BehavioralEvents,
                 trials: TrialEvents
                 ):
        self._info = info
        self._screen = screen
        self._mask = None
        self._data = time_sorted_data
        self._gaze = gaze
        self._behavior = behavior
        self._trials = trials

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

    def processed_fixations(self) -> list:
        """
        Returns a list of dictionaries containing processed fixations for sections of each trial,
        full trial, sample and comparison, along with corresponding trial data.
        """
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

            return fixations_over_letters, switchings

        def process_fixations_over_words(
                fixations : tuple,
                reference_word : str,
                word_shapes_dict : dict,
                relation_letter : str
            ) -> list:

            processed_fixations, fixations_timestamps = fixations

            if processed_fixations.empty:
                return [], {}
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

                # handle 'C' words simultaneous to 'B' figures
                if relation_letter == 'C':
                    if '_' in word:
                       pass # in BC relations, '_' means 'B', so we cannot process letters
                    else:
                        formated_word['letter_fixations'] = process_fixations_over_letters(
                            letters=list(word),
                            processed_fixations=processed_fixations,
                            fixations_timestamps=fixations_timestamps,
                            word_rectangle=word_rectangle
                        )

                elif relation_letter == 'B':
                    if '_' in word: # in CB relations, '_' means 'C', so we can process letters
                        formated_word['letter_fixations'] = process_fixations_over_letters(
                            letters=list(word.replace('_', '')),
                            processed_fixations=processed_fixations,
                            fixations_timestamps=fixations_timestamps,
                            word_rectangle=word_rectangle
                        )

                fixations_over_words.append(formated_word)

            word_switchings = process_switchings(
                fixations_over_words,
                processed_fixations,
                fixations_timestamps)

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

        denormalize = np.vectorize(self._screen.denormalize)
        processed_fixations = process_fixations(self.gaze_df)

        trials = []
        for trial_uid, trial_events in self.events_grouped_by_trial:
            self._trials.to_uid(trial_uid, 'Cycle.ID', self._info.Cycle)
            trial = self._trials.from_uid(trial_uid)
            if trial is None:
                continue
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
                'data': trial
            }

            trials.append(trial_dict)

        return trials

    @property
    def DataFrame(self) -> pd.DataFrame:
        """
        Returns a dataframe with trial data per row.
        """
        def get_sample_figure(trial_events: pd.DataFrame, relation: str) -> str:
            mk = trial_events[self.event_column].str.match(self.events[relation]['samp']['begin'])
            word_shapes_dict = as_dict(trial_events[mk][self.annotation_column].iloc[0])
            return next(iter(word_shapes_dict))

        if self.events_pattern is None:
            self.setup_target_trial_events()

        self._trials.new_column(f'Sample.Figure', 'U12')
        # self._trials.new_column(f'Sample.Figure.Top', np.float64)
        # self._trials.new_column(f'Sample.Figure.Left', np.float64)
        # self._trials.new_column(f'Sample.Figure.Width', np.float64)
        # self._trials.new_column(f'Sample.Figure.Height', np.float64)

        # for i in range(1, 3):
        #     self._trials.new_column(f'Comparison{i}.Figure.Top', np.float64)


        sections = ['Trial', 'Sample', 'Comparisons']
        for section in sections:
            self._trials.new_column(f'{section}.Duration')

        for trial_uid, trial_events in self.events_grouped_by_trial:
            self._trials.to_uid(trial_uid, 'Cycle.ID', self._info.Cycle)
            trial = self._trials.from_uid(trial_uid)

            if trial is None:
                continue
            relation = trial['Relation']

            # assign events pattern of sample
            sample_event_pattern = self.get_event_pattern(
                [self.events[relation]['samp']['begin'],
                 self.events[relation]['samp']['end']]
            )

            self._trials.to_uid(trial_uid, 'Sample.Figure', get_sample_figure(trial_events, relation))

            # assign events pattern of comparisons
            if relation == 'CD':
                comparison_event_pattern = None
            else:
                comparison_event_pattern = self.get_event_pattern(
                    [self.events[relation]['comp']['begin'],
                    self.events[relation]['comp']['end']]
                )

            sections = ['Trial', 'Sample', 'Comparisons']
            events = [
                self.events_pattern,
                sample_event_pattern,
                comparison_event_pattern]

            for section, event_pattern in zip(sections, events):
                column = f'{section}.Duration'
                section_duration = None
                if event_pattern is not None:
                    mask = trial_events[self.event_column].str.match(event_pattern)
                    events = trial_events[mask]

                    if len(events) > 0:
                        timestamps = events[self._behavior.__Timestamp__]
                        section_duration = timestamps.max() - timestamps.min()
                        if section_duration == 0:
                            section_duration = None

                self._trials.to_uid(trial_uid, column, section_duration)

        return self._trials.DataFrame


    def raw_fixations(self) -> list:
        """
        Returns a list of dictionaries with raw fixations for both sections of the trial
        sample and comparisons, along with corresponding trial data.
        """
        def process_section(
            fixations : pd.DataFrame,
            trial_events : pd.DataFrame,
            event_pattern : str,
            section : str,
            relation : str
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

            mask = fixations[self._gaze.__Timestamp__].between(timestamps.min(), timestamps.max())
            section_fixations = fixations[mask]

            if section == 'sample':
                mk = trial_events[self.event_column].str.match(self.events[relation]['samp']['begin'])
                word_shapes_dict = as_dict(trial_events[mk][self.annotation_column].iloc[0])

                result['fixations'] = section_fixations
                result['words'] = word_shapes_dict
                result['relation_letter'] = relation[0]

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

                result['fixations'] = section_fixations
                result['word_shapes'] = word_shapes_dict
                result['relation_letter'] = relation[1]

            return result

        # assign events pattern of trial
        if self.events_pattern is None:
            self.setup_target_trial_events()

        denormalize = np.vectorize(self._screen.denormalize)
        processed_fixations = denormalize(self.gaze_df['FPOGX'], self.gaze_df['FPOGY'])
        processed_fixations = pd.DataFrame({
            'FPOGX': processed_fixations[0],
            'FPOGY': processed_fixations[1],
            'TIME_TICK': self.gaze_df['TIME_TICK'],
            'FPOGD': self.gaze_df['FPOGD'],
            'FPOGID': self.gaze_df['FPOGID'],
            'FPOGV': self.gaze_df['FPOGV']
        })

        sections = ['sample', 'comparisons']
        trials = []
        for trial_uid, trial_events in self.events_grouped_by_trial:
            trial = self._trials.from_uid(trial_uid)
            if trial is None:
                continue
            relation = trial['Relation']

            # assign events pattern of samples
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

            events = [
                sample_event_pattern,
                comparison_event_pattern]

            data_by_section = {}
            for section, event_pattern in zip(sections, events):
                if event_pattern is None:
                    data_by_section[section] = None
                    continue

                data_by_section[section] = process_section(
                    fixations=processed_fixations,
                    trial_events=trial_events,
                    event_pattern=event_pattern,
                    section=section,
                    relation=relation,
                )

            trial_dict = {
                'sections': data_by_section,
                'data': trial
            }

            trials.append(trial_dict)

        return trials


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
        self.trials.label = 2
        self.trials_indices = self.trials.indices

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
        data3 = np.array(list(zip(
            self.trials.timestamps,
            self.trials.labels,
            self.trials.indices)), dtype=dtype)

        self._data = np.concatenate((data1, data2, data3))
        # self._data = np.concatenate((data1, data2))
        self._data.sort(order='timestamp')
        self.per_second = TimeFilter(self._data, 1.0)
        self.per_frame = TimeFilter(self._data, 1/self.desired_fps)
        self.duration = self._get_duration()

    def _get_duration(self):
        return np.max((self.gaze.duration(), self.behavior.duration()))

    def trial_filter(self) -> TrialFilter:
        return TrialFilter(
            info=self.info,
            screen=self.screen,
            time_sorted_data=self._data,
            gaze=self.gaze,
            behavior=self.behavior,
            trials=self.trials)

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