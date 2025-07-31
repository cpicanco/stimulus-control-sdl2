import pandas as pd
import numpy as np

def load_from_file(filename, dtype, converters) -> np.ndarray:
    try:
        return np.loadtxt(filename,
                          delimiter='\t',
                          dtype=dtype,
                          encoding='utf-8',
                          skiprows=1,
                          converters=converters)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")

class BaseEvents:
    __extension__ = ''
    __Timestamp__ = ''
    __dtype__ = None
    __converters__ = None

    def __init__(self, info, version='1'):
        self._label = None
        self.events = None
        self.info = info
        self.base_filename = self.info.base_filename
        self.__load_from_file(version)

    def __load_from_file(self, version=''):
        extension = self.__class__.__extension__
        filename = self.base_filename + extension
        if version == 'TrialEvents1':
            dtype = {
                'names': self.__class__.__dtype__['names'][:-1],
                'formats': self.__class__.__dtype__['formats'][:-1]}
        else:
            dtype = self.__class__.__dtype__
        converters = self.__class__.__converters__
        self.events = load_from_file(filename, dtype, converters)

    def duration(self):
        if self.events.size == 0:
            return 0
        return self.events[self.__class__.__Timestamp__][-1]

    @property
    def timestamps(self) -> np.ndarray:
        return self.events[self.__class__.__Timestamp__]

    @property
    def indices(self):
        if self.events is None:
            return np.array([])
        return np.arange(len(self.events[self.__class__.__Timestamp__]))

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: int):
        self._label = label

    @property
    def labels(self):
        if self.events is None:
            return np.array([])
        return np.full(len(self.events[self.__class__.__Timestamp__]), self._label, dtype=np.uint8)

    @property
    def DataFrame(self):
        if self.events is None:
            return pd.DataFrame()
        return pd.DataFrame(self.events)