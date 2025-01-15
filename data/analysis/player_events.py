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

    def __init__(self, info):
        self.events = None
        self.info = info
        self.base_filename = self.info.base_filename
        self._label = None
        self.__load_from_file()

    def __load_from_file(self):
        extension = self.__class__.__extension__
        filename = self.base_filename + extension
        dtype = self.__class__.__dtype__
        converters = self.__class__.__converters__
        self.events = load_from_file(filename, dtype, converters)

    def duration(self):
        return self.events[self.__class__.__Timestamp__][-1]

    @property
    def timestamps(self) -> np.ndarray:
        return self.events[self.__class__.__Timestamp__]

    @property
    def indices(self):
        return np.arange(len(self.events[self.__class__.__Timestamp__]))

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: int):
        self._label = label

    @property
    def labels(self):
        return np.full(len(self.events[self.__class__.__Timestamp__]), self._label, dtype=np.uint8)

    @property
    def DataFrame(self):
        return pd.DataFrame(self.events)