import cv2
from player_constants import DEFAULT_FPS

class FPS:
    def __init__(self):
        self.value = DEFAULT_FPS

class Recorder:
    def __init__(self,
                 width=1440,
                 height=900,
                 video_filename='drawing.mp4',
                 fps=FPS()):
        self.video_filename = video_filename
        self.size = (width, height)
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.device = cv2.VideoWriter()

    def start(self):
        if not self.device.isOpened():
            self.device.open(self.video_filename, self.fourcc, self.fps.value, self.size)
            self.after_start()
            print(f"Recording started, saving to '{self.video_filename}'")

    def after_start(self):
        pass

    def before_stop(self):
        pass

    def stop(self):
        if self.device.isOpened():
            self.before_stop()
            self.device.release()
            print(f"Recording saved.")

    def write(self, frame):
        self.device.write(frame)

if __name__ == '__main__':
    import numpy as np
    from player_media import arimo

    width = 1440
    height = 900
    channels = 3
    frame = np.zeros((height, width, channels), dtype=np.uint8)
    recorder = Recorder(width=width, height=height,
                        video_filename='drawing.mp4')
    recorder.start()
    if not recorder.device.isOpened():
        raise RuntimeError("VideoWriter could not be opened.")
    for i in range(0, 100):
        ft.putText(img=frame,
                text='Quick Fox',
                org=(15, 70),
                fontHeight=60,
                color=(255,  255, 255),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=True)
        recorder.write(frame)
    recorder.stop()