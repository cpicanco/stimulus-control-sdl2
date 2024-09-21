import numpy as np
from player_events import BaseEvents

class GazeEvents(BaseEvents):
    __extension__ = '.gaze'
    __Count__ = 'CNT'
    __Timestamp__ = 'TIME_TICK'
    __FixationX__ = 'FPOGX'
    __FixationY__ = 'FPOGY'
    __FixationStart__ = 'FPOGS'
    __FixationDuration__ = 'FPOGD'
    __FixationID__ = 'FPOGID'
    __FixationValid__ = 'FPOGV'
    __LeftGazeX__ = 'LPOGX'
    __LeftGazeY__ = 'LPOGY'
    __LeftGazeValid__ = 'LPOGV'
    __RightGazeX__ = 'RPOGX'
    __RightGazeY__ = 'RPOGY'
    __RightGazeValid__ = 'RPOGV'
    __BestGazeX__ = 'BPOGX'
    __BestGazeY__ = 'BPOGY'
    __BestGazeValid__ = 'BPOGV'
    __dtype__ = {
        'names' : (
            __Count__,
            __Timestamp__,
            __FixationX__,
            __FixationY__,
            __FixationStart__,
            __FixationDuration__,
            __FixationID__,
            __FixationValid__,
            __LeftGazeX__,
            __LeftGazeY__,
            __LeftGazeValid__,
            __RightGazeX__,
            __RightGazeY__,
            __RightGazeValid__,
            __BestGazeX__,
            __BestGazeY__,
            __BestGazeValid__
        ),
        'formats' : (
            np.uint64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.uint64,
            np.bool_,
            np.float64,
            np.float64,
            np.bool_,
            np.float64,
            np.float64,
            np.bool_,
            np.float64,
            np.float64,
            np.bool_
    )}
    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def missing_events(self):
        """
        Returns the number of missing events.
        """
        # subtract all counts from the first count
        count = self.events[self.__Count__]
        count = count - count.min() + 1
        # subtract the length from the last count
        return int(count.size - count[-1])

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__][0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] / self.info.TimeTickFrequency

if __name__ == '__main__':
    from player_information import GazeInfo
    from fileutils import cd, data_dir

    data_dir()
    cd('26-MSS')
    cd('analysis')
    cd('2024-08-29')
    cd('4')
    info = GazeInfo('030')

    gaze = GazeEvents(info)
    print(gaze.events[gaze.__BestGazeValid__])
    print(f'{gaze.missing_events()} missing events')
    print('duration:', gaze.duration())

    data_dir()