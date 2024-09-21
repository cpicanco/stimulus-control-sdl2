from PySide6.QtCore import (
    QPoint,
    QSize,
)

from PySide6.QtGui import (
    QPainter,
)

from PySide6.QtWidgets import (
    QWidget
)

from player_drawings import (
    Bitmap,
    Circle,
    Empty
)

from player_recorder import (
    Recorder
)

class RenderArea(QWidget):
    def __init__(self, parent=None, width=800, height=600):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.setMouseTracking(True)
        self.recorder = Recorder(
                width=width,
                height=height,
                invalidate_callback=self.invalidate,
                video_filename='drawing.mp4')
        self.bitmap = Bitmap(self.width, self.height)

        self.background = Empty(self)
        self.gaze = Circle(self)
        self.components = [self.background, self.gaze]

    def mousePressEvent(self, event):
        self.gaze.visible = not self.gaze.visible
        self.update()

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        self.gaze.x = point.x()
        self.gaze.y = point.y()
        self.update()

    def minimumSizeHint(self):
        return QSize(800, 600)

    def sizeHint(self):
        return QSize(self.width, self.height)

    def paintEvent(self, event):
        canvas = QPainter(self)

        data = self.bitmap.new()
        for child in self.components:
            child.paint(data)
        self.bitmap.data = data

        canvas.drawPixmap(QPoint(0, 0), self.bitmap.pixmap)
        canvas.end()

    def invalidate(self):
        self.recorder.put(self.bitmap.data)

if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = RenderArea()
    window.show()
    sys.exit(app.exec())