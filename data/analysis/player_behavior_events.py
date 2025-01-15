import numpy as np
import pandas as pd

from player_events import BaseEvents

class BehavioralEvents(BaseEvents):
    __extension__ = '.timestamps'
    __Timestamp__ = 'Timestamp'
    __Trial__ = 'Session.Trial.UID'
    __Block__ = 'Session.Block.UID'
    __Event__ = 'Event'
    __Annotation__ = 'Event.Annotation'
    __dtype__ = { 'names' : (__Timestamp__, __Trial__, __Block__, __Event__, __Annotation__),
                  'formats' : (np.float64, np.uint64, np.uint64, 'U40', 'U100') }
    __converters__ = {0 : lambda x: x.replace(',', '.')}

    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__][0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__]

class TrialEvents(BaseEvents):
    __extension__ = '.data.processed'
    __Timestamp__ = 'Report.Timestamp'
    __Trial__ = 'Session.Trial.UID'
    __Block__ = 'Session.Block.UID'
    __BlockTrial__ = 'Session.Block.Trial.UID'
    __BlockID__ = 'Session.Block.ID'
    __BlockTrialID__ = 'Session.Block.Trial.ID'
    __BlockName__ = 'Session.Block.Name'
    __TrialID__ = 'Trial.ID'
    __Cycle_ID = 'Cycle.ID'
    __Name__ = 'Name'
    __Relation = 'Relation'
    __Comparisons__ = 'Comparisons'
    __Result = 'Result'
    __CounterHit__ = 'CounterHit'
    __CounterHitMaxConsecutives__ = 'CounterHit.MaxConsecutives'
    __CounterMiss__ = 'CounterMiss'
    __CounterMissMaxConsecutives__ = 'CounterMiss.MaxConsecutives'
    __CounterNone__ = 'CounterNone'
    __CounterNoneMaxConsecutives__ = 'CounterNone.MaxConsecutives'
    __SamplePosition1__ = 'Sample-Position.1'
    __ComparisonPosition1__ = 'Comparison-Position.1'
    __ComparisonPosition2__ = 'Comparison-Position.2'
    __ComparisonPosition3__ = 'Comparison-Position.3'
    __Response__ = 'Response'
    __HasDifferentialReinforcement__ = 'HasDifferentialReinforcement'
    __Latency__ = 'Latency'
    __Participant__ = 'Participant'
    __Condition__ = 'Condition'
    __Date__ = 'Date'
    __Time__ = 'Time'
    __File__ = 'File'
    __dtype__ = {
        'names' : (
            __Timestamp__,                      # np.float64
            __Trial__,                          # np.uint64
            __Block__,                          # np.uint64
            __BlockTrial__,                     # np.uint64
            __BlockID__,                        # np.uint64
            __BlockTrialID__,                   # np.uint64
            __BlockName__,                      # U40
            __TrialID__,                        # np.uint64
            __Cycle_ID,                         # np.uint8
            __Name__,                           # U40
            __Relation,                         # U40
            __Comparisons__,                    # np.uint8
            __Result,                           # np.bool_
            __CounterHit__,                     # np.uint64
            __CounterHitMaxConsecutives__,      # np.uint64
            __CounterMiss__,                    # np.uint64
            __CounterMissMaxConsecutives__,     # np.uint64
            __CounterNone__,                    # np.uint64
            __CounterNoneMaxConsecutives__,     # np.uint64
            __SamplePosition1__,                # U20
            __ComparisonPosition1__,            # U20
            __ComparisonPosition2__,            # U20
            __ComparisonPosition3__,            # U20
            __Response__,                       # U20
            __HasDifferentialReinforcement__,   # np.bool_
            __Latency__,                        # np.float64
            __Participant__,                    # U10
            __Condition__,                      # np.uint8
            __Date__,                           # U10
            __Time__,                           # U10
            __File__                            # U20
        ),
        'formats' : (
            np.float64,
            np.uint64,
            np.uint64,
            np.uint64,
            np.uint64,
            np.uint64,
            'U40',
            np.uint64,
            np.uint8,
            'U40',
            'U40',
            np.uint8,
            np.uint8,
            np.uint64,
            np.uint64,
            np.uint64,
            np.uint64,
            np.uint64,
            np.uint64,
            'U20',
            'U20',
            'U20',
            'U20',
            'U20',
            np.bool_,
            np.float64,
            'U10',
            'U4',
            'U10',
            'U10',
            'U20'
            ) }
    __converters__ = {
        0 : lambda x: x.replace(',', '.'),
        10 : lambda x: x.replace('-', ''),
        12 : lambda x: x.replace('Hit', '1').replace('Miss', '0'),
        24 : lambda x: x.replace('True', '1').replace('False', '0'),
        25 : lambda x: x.replace(',', '.')}

    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__][0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__]

    def from_uid(self, uid: int) -> dict:
        row = self.events[self.events[self.__Trial__] == uid][0]
        return dict(zip(row.dtype.names, row))

if __name__ == '__main__':
    from player_information import GazeInfo
    from fileutils import cd, data_dir

    data_dir()
    cd('5-NUN')
    cd('analysis')
    cd('2024-08-19')
    cd('1')
    info = GazeInfo('003')

    events = BehavioralEvents(info)
    print('duration:', events.duration())
    for event in events.events[events.__Annotation__]:
        print(event)

    events = TrialEvents(info)
    print(events.from_uid(1))
    data_dir()