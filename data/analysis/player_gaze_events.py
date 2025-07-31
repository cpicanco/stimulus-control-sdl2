import numpy as np
from player_events import BaseEvents
from player_eye_math import SCREEN_W_CM, SCREEN_H_CM

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
        if self.events is None:
            self.events = np.array([], dtype=self.__dtype__)
            return
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

    def denormalize(self, measurement='best_gaze', metric='centimeter'):
        if measurement == 'best_gaze':
            m_x = self.__BestGazeX__
            m_y = self.__BestGazeY__
        elif measurement == 'left_gaze':
            m_x = self.__LeftGazeX__
            m_y = self.__LeftGazeY__
        elif measurement == 'right_gaze':
            m_x = self.__RightGazeX__
            m_y = self.__RightGazeY__
        elif measurement == 'fixations':
            m_x = self.__FixationX__
            m_y = self.__FixationY__

        if metric == 'pixel':
            x = self.events[m_x] * self.info.ScreenWidth
            y = self.events[m_y] * self.info.ScreenHeight
        elif metric == 'centimeter':
            x = self.events[m_x] * SCREEN_W_CM
            y = self.events[m_y] * SCREEN_H_CM
        return (x, y)

if __name__ == '__main__':
    import os
    from player_information import GazeInfo
    from fileutils import cd, data_dir

    data_dir()
    filepath = os.path.join('estudo2', '12-HUM', 'analysis', '2024-07-24', '1')
    filename = '001'

    cd(filepath)
    info = GazeInfo(filename)

    gaze = GazeEvents(info)
    print(gaze.events[gaze.__FixationValid__][0])
    print(gaze.events[gaze.__FixationValid__][7])
    # print(f'{gaze.missing_events()} missing events')
    # print('duration:', gaze.duration())