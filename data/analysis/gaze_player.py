import ast
import re

import pandas as pd
from player_screen_recorder import Device as OpenCVDevice

def quote_keys(dict_string):
    # Use regular expressions to add quotes around the keys
    quoted_string = re.sub(r'(\w+):', r'"\1":', dict_string)
    return quoted_string

def load_from_file(filename):
    try:
        if filename.endswith(GazeInfo.__gaze_info_extension__) \
        or filename.endswith(GazeInfo.__info_extension__):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.readlines()

        elif filename.endswith(StimuliPlayer.__extension__) \
        or filename.endswith(GazePlayer.__extension__):
            return pd.read_csv(filename, sep='\t', encoding='utf-8', engine='python')
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")


class GazeInfo:
    __info_extension__ = '.info.processed'
    __gaze_info_extension__ = '.gaze.info'

    __ClientVersion__ = 'ClientVersion'
    __TimeTickFrequency__ = 'TimeTickFrequency'
    __CameraWidth__ = 'CameraWidth'
    __CameraHeight__ = 'CameraHeight'
    __ProductID__ = 'ProductID'
    __ProductBus__ = 'ProductBus'
    __ProductRate__ = 'ProductRate'
    __SerialID__ = 'SerialID'
    __CompanyID__ = 'CompanyID'
    __APIID__ = 'APIID'
    __Monitor__ = 'Monitor'
    __ScreenWidth__ = 'ScreenWidth'
    __ScreenHeight__ = 'ScreenHeight'

    def __init__(self, base_filename):
        self.base_filename = base_filename
        self.ClientVersion = None
        self.TimeTickFrequency = None
        self.CameraWidth = None
        self.CameraHeight = None
        self.ProductID = None
        self.ProductBus = None
        self.ProductRate = None
        self.SerialID = None
        self.CompanyID = None
        self.APIID = None
        self.ScreenWidth = None
        self.ScreenHeight = None
        self.load_from_file()

    def load_from_file(self):
        info_filename = self.base_filename + self.__info_extension__
        gaze_info_filename = self.base_filename + self.__gaze_info_extension__

        lines = load_from_file(info_filename)
        lines += load_from_file(gaze_info_filename)

        for line in lines:
            if line.startswith(self.__ClientVersion__):
                self.ClientVersion = int(line.split(':')[1].strip())
            elif line.startswith(self.__TimeTickFrequency__):
                self.TimeTickFrequency = int(line.split(':')[1].strip())
            elif line.startswith(self.__CameraWidth__):
                self.CameraWidth = int(line.split(':')[1].strip())
            elif line.startswith(self.__CameraHeight__):
                self.CameraHeight = int(line.split(':')[1].strip())
            elif line.startswith(self.__ProductID__):
                self.ProductID = line.split(':')[1].strip()
            elif line.startswith(self.__ProductBus__):
                self.ProductBus = line.split(':')[1].strip()
            elif line.startswith(self.__ProductRate__):
                self.ProductRate = int(line.split(':')[1].strip())
            elif line.startswith(self.__SerialID__):
                self.SerialID = line.split(':')[1].strip()
            elif line.startswith(self.__CompanyID__):
                self.CompanyID = line.split(':')[1].strip()
            elif line.startswith(self.__APIID__):
                self.APIID = float(line.split(':')[1].strip())
            elif line.startswith(self.__Monitor__):
                monitor = line.split(self.__Monitor__+':')[1].strip()
                monitor = ast.literal_eval(quote_keys(monitor))
                # extract screen width and height
                self.ScreenWidth = int(monitor['w'])
                self.ScreenHeight = int(monitor['h'])

class BasePlayer:
    __extension__ = ''
    __Timestamp__ = ''

    def __init__(self, info):
        self.extension = self.__class__.__extension__
        self.info = info
        self.base_filename = self.info.base_filename
        self.__load_from_file()

    def __load_from_file(self):
        self.events = load_from_file(self.base_filename + self.extension)

    def duration(self):
        return self.events[self.__class__.__Timestamp__].iloc[-1]

class StimuliPlayer(BasePlayer):
    __extension__ = '.timestamps'
    __Timestamp__ = 'Timestamp'
    __Trial__ = 'Session.Trial.UID'
    __Block__ = 'Session.Block.UID'
    __Event__ = 'Event'
    __Annotation = 'Event.Annotation'

    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def __convert_timestamps_to_seconds(self):
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__].str.replace(',', '.').astype(float)
        first_timestamp = self.events[self.__Timestamp__].iloc[0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp

class GazePlayer(BasePlayer):
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

    def __init__(self, info):
        super().__init__(info)
        self.__convert_timestamps_to_seconds()

    def missing_events(self):
        """
        Returns the number of missing events.
        """
        # subtract all counts from the first count
        count = self.events[self.__Count__]
        count = count - count.iloc[0] + 1
        # subtract the length from the last count
        return len(count) - count.iloc[-1]

    def __convert_timestamps_to_seconds(self):
        first_timestamp = self.events[self.__Timestamp__].iloc[0]
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] - first_timestamp
        self.events[self.__Timestamp__] = self.events[self.__Timestamp__] / self.info.TimeTickFrequency


    def start(self):
        pass

    def stop(self):
        pass

class Player:
    def __init__(self, info):
        self.info = info
        self.gaze = GazePlayer(info)
        self.stimuli = StimuliPlayer(info)
        self.ui = Device(bitmap_size=(512, 512))

    def show(self):
        pass


if __name__ == '__main__':
    from fileutils import cd, data_dir

    data_dir()
    cd('1-JOP')
    cd('analysis')
    cd('2024-07-22')
    cd('0')
    info = GazeInfo('000')
    gaze = GazePlayer(info)
    stimuli = StimuliPlayer(info)

    print(gaze.missing_events())


    print('Stimuli:', stimuli.duration())
    print('Gaze:', gaze.duration())

    data_dir()