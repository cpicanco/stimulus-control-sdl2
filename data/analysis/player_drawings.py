import threading
from typing import Self

from PySide6.QtGui import (
    QImage,
    QPixmap
)

import cv2
import numpy as np

from player_media import (
    load_pictures,
    load_instructions,
    load_words
)

from player_utils import (
    as_dict
)

class Bitmap:
    def __init__(self, width, height, channels=3, has_qt_pixmap=True):
        self.lock = threading.Lock()
        self.width = width
        self.height = height
        self.channels = channels
        self._data = self.new()
        if has_qt_pixmap:
            self.pixmap = self.pixmap_from_data()
        else:
            self.pixmap = None

    def new(self):
        return np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

    @property
    def data(self):
        self.lock.acquire()
        data = self._data.copy()
        self.pixmap = self.pixmap_from_data()
        self.lock.release()
        return data

    @data.setter
    def data(self, data):
        self.lock.acquire()
        self._data = data
        self.lock.release()
        self.invalidate

    @property
    def invalidate(self):
        self.pixmap = self.pixmap_from_data()

    @property
    def clear(self):
        self.lock.acquire()
        self._data.fill(0)
        self.lock.release()

    def data_from_pixmap(self):
        return np.frombuffer(
            self.pixmap.toImage().constBits(),
            dtype = np.uint8).reshape((self.height, self.width, self.channels))

    def data_from_pixmap_RGBA(self):
        return cv2.cvtColor(np.frombuffer(
            self.pixmap.toImage().constBits(),
            dtype = np.uint8).reshape((self.height, self.width, self.channels)),
            cv2.COLOR_RGBA2RGB)

    def pixmap_from_data(self):
        return QPixmap.fromImage(
            QImage(
                self._data,
                self.width,
                self.height,
                self.width * self.channels,
                QImage.Format_BGR888))


class Component:
    def __init__(self, parent=None):
        self.parent = parent
        self.x = 0
        self.y = 0
        self._visible = False
        self.children : list[Self] = []
        self.image = None

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value


    def load(self, key, bounds_rect):
        pass

    def draw(self, buffer):
        image = self.image
        width = image.shape[1]
        height = image.shape[0]
        # how to overlay a heatmap on top of the image?
        buffer[self.y:self.y + height, self.x:self.x + width] = image

    def paint(self, buffer):
        if self.visible:
            self.draw(buffer)

class Screen(Component):
    def __init__(self, width, height):
        super().__init__(None)
        self.width = width
        self.height = height

    def denormalize_x(self, value):
        return round(value*self.width)

    def denormalize_y(self, value):
        return round(value*self.height)

    def denormalize(self, x, y):
        return self.denormalize_x(x), self.denormalize_y(y)

screen = Screen(1440, 900)

class Circle(Component):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = 0
        self.y = 0
        self.radius = 25
        self.color = (0, 0, 255)
        self.length = 2

    @property
    def visible(self):
        if self.parent.x <= self.x <= self.parent.width \
        and self.parent.y <= self.y <= self.parent.height:
            return self._visible
        else:
            return False

    @visible.setter
    def visible(self, value):
        self._visible = value

    def draw(self, buffer):
        cv2.circle(
            buffer,
            (self.x, self.y),
            self.radius,
            self.color,
            self.length,
            cv2.LINE_AA)

class Rectangle(Component):
    def __init__(self, parent=None, width=25, height=25, color=(0, 0, 0), length=1):
        super().__init__(parent)
        self.x = 0
        self.y = 0
        self.width = width
        self.height = height
        self.color = color
        self.length = length

    def draw(self, buffer):
        cv2.rectangle(
            buffer,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            self.color,
            self.length,
            cv2.LINE_AA)

class Background(Component):
    def __init__(self, parent=None, color=(0, 0, 0)):
        super().__init__(parent)
        self.color = color
        self.enabled = False

    def draw(self, buffer):
        cv2.rectangle(
            buffer,
            (0, 0),
            (buffer.shape[1], buffer.shape[0]),
            self.color,
            -1,
            cv2.LINE_AA
        )

class Nothing(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

    def draw(self, buffer):
        pass

class Speech(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

class Word(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

    def load(self, key, bounds_rect):
        self.x = bounds_rect['x']
        self.y = bounds_rect['y']
        self.image = self.parent.words[key]


class Instruction(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

    def load(self, key, bounds_rect):
        self.x = bounds_rect['x']
        self.y = bounds_rect['y']
        self.image = self.parent.instructions[key]

class Picture(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

    def load(self, key, bounds_rect):
        self.x = bounds_rect['x']
        self.y = bounds_rect['y']
        self.image = self.parent.pictures[key]

class Calibration(Component):
    def __init__(self, parent=None):
        super().__init__(parent)

    def load(self, key, bounds_rect):
        pass

    def draw(self, buffer):
        pass

ModalityDict = {
    'A': Picture,
    'B': Picture,
    'C': Word,
    'D': Speech
}

class MTS(Component):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.sample_modality = None
        self.comparisons_modality = None
        self.sample = None
        self.comparisons : None

    def load(self, event, event_annotation):
        if event.startswith('MTSStimuli') and event.endswith('.Show'):
            self.visible = True
            annotation : dict = as_dict(event_annotation)
            e = event.split('.')
            modality = e[1].replace('Modality', '')
            self.sample_modality = modality[0]
            self.comparisons_modality = modality[1]

            sample_class = ModalityDict[modality[0]]
            self.sample = sample_class(self.parent)
            for key, value in annotation.items():
                self.sample.load(key, value)
            self.sample.visible = True
            self.children.append(self.sample)

        elif event.startswith('Start.Audio.Sample'): # sample stopped
            if self.sample_modality == 'A':
                self.sample.visible = False

        elif event.startswith('Comparisons.Start'):
            annotation : dict = as_dict(event_annotation)
            comparison_class = ModalityDict[self.comparisons_modality]
            self.comparisons = []
            for key, value in annotation.items():
                comparison = comparison_class(self.parent)
                comparison.load(key, value)
                comparison.visible = True
                self.comparisons.append(comparison)
                self.children.append(comparison)

        if event.startswith('MTSStimuli') and event.endswith('.Hide'):
            self.children = []
            self.visible = False

    def paint(self, buffer):
        for child in self.children:
            child.paint(buffer)

class BehaviorDrawingFactory:
    def __init__(self, participant_id):
        self.border = Rectangle(parent=self)
        self.border.left = 0
        self.border.top = 0
        self.border.width = screen.width
        self.border.height = screen.height
        self.border.length = 4

        self.timeout = Background(parent=self, color=(0, 0, 60))
        self.calibration = Calibration(parent=self)
        self.instruction = Instruction(parent=self)
        self.mts = MTS(parent=self)
        self.ending = Word(parent=self)

        self.pictures = load_pictures(participant_id)
        self.instructions = load_instructions()
        self.words = load_words()

        self.components = [
            self.border,
            self.timeout,
            self.calibration,
            self.instruction,
            self.mts,
            self.ending
        ]

    def update(self, line):
        event: str = line['Event'].decode('utf-8')
        event_annotation: str = line['Event.Annotation'].decode('utf-8')

        if event.startswith('CalibrationStimuli'):
            # self.calibration.load(key, bounds_rect)
            pass

        elif event.startswith('InstructionStimuli'):
            annotation = as_dict(event_annotation)
            for key, value in annotation.items():
                self.instruction.visible = True
                self.instruction.load(key, value)

        elif event.startswith('LastStimuli.Show'):
            key = 'Fim.'
            value = {'x':577,'y':362}
            self.ending.visible = True
            self.ending.load(key, value)

        elif event.startswith('MTSStimuli') \
          or event.startswith('Start.Audio.Sample') \
          or event.startswith('Comparisons.Start'):
            self.instruction.visible = False
            self.mts.load(event, event_annotation)

        elif event.startswith('Miss.Stop'):
            self.timeout.enabled = True

        if event.startswith('MTSStimuli') \
         and event.endswith('.Show'):
            if self.timeout.enabled:
                self.timeout.enabled = False
                self.timeout.visible = False

        elif event.startswith('MTSStimuli') \
         and event.endswith('.Hide'):
            if self.timeout.enabled:
                self.timeout.visible = True

    @property
    def visible_components(self):
        return [component for component in self.components if component.visible]


class GazeDrawingFactory:
    def __init__(self, session, screen : Screen):
        self.x_label = session.gaze.__FixationX__
        self.y_label = session.gaze.__FixationY__
        self.v_label = session.gaze.__FixationValid__
        # self.x_label = session.gaze.__BestGazeX__
        # self.y_label = session.gaze.__BestGazeY__
        # self.v_label = session.gaze.__BestGazeValid__

        # self.fx_label = session.gaze.__FixationX__
        # self.fy_label = session.gaze.__FixationY__
        # self.fs_label = session.gaze.__FixationStart__
        # self.fd_label = session.gaze.__FixationDuration__
        # self.fv_label = session.gaze.__FixationValid__

        self.screen = screen
        self.gaze = Circle(parent=self.screen)

    def update(self, line):
        self.gaze.visible = line[self.v_label]
        # self.gaze.visible = True
        self.gaze.x, self.gaze.y = self.screen.denormalize(line[self.x_label], line[self.y_label])

    def paint(self, img):
        self.gaze.paint(img)

if __name__ == '__main__':
    word = 'nela'
    data = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.putText(data, word, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('test', data)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()