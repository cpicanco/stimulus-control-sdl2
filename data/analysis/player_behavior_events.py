import numpy as np

from player_events import BaseEvents

class BehavioralEvents(BaseEvents):
    __extension__ = '.timestamps'
    __Timestamp__ = 'Timestamp'
    __Trial__ = 'Session.Trial.UID'
    __Block__ = 'Session.Block.UID'
    __Event__ = 'Event'
    __Annotation__ = 'Event.Annotation'
    __dtype__ = { 'names' : (__Timestamp__, __Trial__, __Block__, __Event__, __Annotation__),
                  'formats' : (np.float64, np.uint64, np.uint64, 'S40', 'S100') }
    __converters__ = {0 : lambda x: x.replace(',', '.')}

    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__][0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__]

if __name__ == '__main__':
    from player_information import GazeInfo
    from fileutils import cd, data_dir

    data_dir()
    cd('26-MSS')
    cd('analysis')
    cd('2024-08-29')
    cd('4')
    info = GazeInfo('030')

    events = BehavioralEvents(info)
    print(events.events[events.__Annotation__][-1].decode('utf-8'))
    print('duration:', events.duration())

    data_dir()