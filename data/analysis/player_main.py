from PySide6.QtCore import (
    __version__,
    Qt
)

from PySide6.QtWidgets import (
    QGridLayout,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
)


import cv2
import numpy as np
from player_renderer import RenderArea

class Window(RenderArea):
    def __init__(self, width=800, height=600):
        super().__init__(width=width, height=height)
        button_start_recording = QPushButton("&Start Recording", self)
        button_stop_recording = QPushButton("&Stop Recording", self)

        button_start_recording.clicked.connect(self.recorder.start)
        button_stop_recording.clicked.connect(self.recorder.stop)

        vspace = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding)

        # layout = QVBoxLayout(self)
        # layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)
        # layout.alignment = Qt.AlignTop
        # layout.addWidget(button_start_recording)
        # layout.addWidget(button_stop_recording)

        layout = QGridLayout(self)
        layout.alignment = Qt.AlignTop
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setColumnMinimumWidth(0, int(width*0.8))

        layout.addWidget(button_start_recording, layout.rowCount(), 1, alignment=Qt.AlignRight)
        layout.addWidget(button_stop_recording, layout.rowCount()+1, 1, alignment=Qt.AlignRight)
        layout.addItem(vspace, layout.rowCount()+1, 1)

        self.setLayout(layout)
        self.setWindowTitle("Player")
        print('OpenCV version: {}'.format(cv2.__version__))
        print('Numpy version: {}'.format(np.__version__))
        print('Qt version: {}'.format(__version__))

if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())