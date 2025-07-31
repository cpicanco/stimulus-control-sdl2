import numpy as np
from numpy.lib.recfunctions import append_fields

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
    __CycleID__ = 'Cycle.ID'
    __Name__ = 'Name'
    __Relation__ = 'Relation'
    __Comparisons__ = 'Comparisons'
    __Result__ = 'Result'
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
    __Design__ = 'Design'
    __dtype__ = {
        'names' : (
            __Timestamp__,                      # 0
            __Trial__,                          # 1
            __Block__,                          # 2
            __BlockTrial__,                     # 3
            __BlockID__,                        # 4
            __BlockTrialID__,                   # 5
            __BlockName__,                      # 6
            __TrialID__,                        # 7
            __CycleID__,                        # 8
            __Name__,                           # 9
            __Relation__,                       # 10
            __Comparisons__,                    # 11
            __Result__,                         # 12
            __CounterHit__,                     # 13
            __CounterHitMaxConsecutives__,      # 14
            __CounterMiss__,                    # 15
            __CounterMissMaxConsecutives__,     # 16
            __CounterNone__,                    # 17
            __CounterNoneMaxConsecutives__,     # 18
            __SamplePosition1__,                # 19
            __ComparisonPosition1__,            # 20
            __ComparisonPosition2__,            # 21
            __ComparisonPosition3__,            # 22
            __Response__,                       # 23
            __HasDifferentialReinforcement__,   # 24
            __Latency__,                        # 25
            __Participant__,                    # 26
            __Condition__,                      # 27
            __Date__,                           # 28
            __Time__,                           # 29
            __File__,                           # 30
            __Design__                          # 31
        ),
        'formats' : (
            np.float64, # 0
            np.uint64,  # 1
            np.uint64,  # 2
            np.uint64,  # 3
            np.uint64,  # 4
            np.uint64,  # 5
            'U40',      # 6
            np.uint64,  # 7
            np.uint8,   # 8
            'U40',      # 9
            'U40',      # 10
            np.uint8,   # 11
            np.uint8,   # 12
            np.uint64,  # 13
            np.uint64,  # 14
            np.uint64,  # 15
            np.uint64,  # 16
            np.uint64,  # 17
            np.uint64,  # 18
            'U20',      # 19
            'U20',      # 20
            'U20',      # 21
            'U20',      # 22
            'U20',      # 23
            np.uint8,   # 24
            np.float64, # 25
            'U10',      # 26
            'U4',       # 27
            'U10',      # 28
            'U10',      # 29
            'U20',      # 30
            'U20'       # 31
            ) }
    __converters__ = {
        0 : lambda x: x.replace(',', '.'),
        10 : lambda x: x.replace('-', ''),
        12 : lambda x: x.replace('Hit', '1').replace('Miss', '0').replace('INVALID', '255'),
        23 : lambda x: x.strip().replace('INVALID', 'nan'),
        24 : lambda x: x.replace('True', '1').replace('False', '0'),
        25 : lambda x: x.replace(',', '.')}

    def __init__(self, info, version='1'):
        super().__init__(info, self.__class__.__name__+version)
        self.__convert_timestamps_to_seconds()

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__][0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__]

    def new_column(self, column: str, dtype = np.float64) -> None:
        if column not in self.events.dtype.names:
            default_values = np.full(len(self.events), None, dtype=dtype)
            self.events = append_fields(self.events, column, default_values, usemask=False)

    def from_uid(self, uid: int) -> dict:
        data = self.events[self.events[self.__Trial__] == uid]
        if data.size == 0:
            return None
        else:
            row = data[0]
            return dict(zip(row.dtype.names, row))

    def to_uid(self, uid: int, column: str, value) -> dict:
        mask = self.events[self.__Trial__] == uid
        self.events[column][mask] = value

if __name__ == '__main__':
    from player_information import GazeInfo
    from fileutils import cd, data_dir

    data_dir()
    cd('estudo2')
    cd('12-HUM')
    cd('analysis')
    cd('2024-07-24')
    cd('1')
    info = GazeInfo('001')

    # events = BehavioralEvents(info)
    # print('duration:', events.duration())
    # for event in events.events[events.__Annotation__]:
    #     print(event)

    events = TrialEvents(info)
    print(events.from_uid(1)['HasDifferentialReinforcement'])