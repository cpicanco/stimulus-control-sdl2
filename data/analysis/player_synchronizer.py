import numpy as np

from player_information import GazeInfo
from player_behavior_events import BehavioralEvents
from player_gaze_events import GazeEvents
from player_constants import DEFAULT_FPS

class DataFilter:
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

class Synchronizer:
    def __init__(self, info: GazeInfo):
        self.info = info

        self.behavior = BehavioralEvents(info)
        self.behavior.label = 0
        self.behavior_indices = self.behavior.indices


        self.gaze = GazeEvents(info)
        self.gaze.label = 1
        self.gaze_indices = self.gaze.indices

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
        self.per_second = DataFilter(self._data, 1.0)
        self.per_frame = DataFilter(self._data, 1/self.desired_fps)
        self.duration = self._get_duration()

    def _get_duration(self):
        return np.max((self.gaze.duration(), self.behavior.duration()))

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
    from player_media import load_pictures
    from player_information import GazeInfo
    from fileutils import cd, data_dir
    from player_recorder import Recorder
    from player_drawings import (
        screen,
        BehaviorDrawingFactory,
        GazeDrawingFactory
    )

    participant_id = '26-MSS'
    load_pictures(participant_id)
    data_dir()
    cd(participant_id)
    cd('analysis')
    cd('2024-08-30')
    cd('5')
    code = '040'
    info = GazeInfo(code)
    screen.width = info.ScreenWidth
    screen.height = info.ScreenHeight
    session = Synchronizer(info)

    # bitmap = Bitmap(width=screen.width, height=screen.height, has_qt_pixmap=False)
    recorder = Recorder(
        width=screen.width,
        height=screen.height,
        video_filename=code+'.mp4')
    recorder.start()

    behavior_factory = BehaviorDrawingFactory(participant_id)
    gaze_factory = GazeDrawingFactory(session, screen)

    for time, frame in session.frames:
        # if time < 30:
        #     continue
        print(f'Frame: {time}')
        behavior_data = frame[frame['label'] == 0]
        gaze_data = frame[frame['label'] == 1]
        img = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)

        for timestamp, _, i in behavior_data:
            behavior_factory.update(session.behavior.events[i])

        for component in behavior_factory.visible_components:
            component.paint(img)

        for timestamp, _, i in gaze_data:
            gaze_factory.update(session.gaze.events[i])
            gaze_factory.paint(img)

        recorder.device.write(img)
        # if time > 50:
        #     break

    recorder.stop()