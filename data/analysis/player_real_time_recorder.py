
import time
from queue import Queue
import threading

import cv2

from player_constants import DEFAULT_FPS, BUFFER_SIZE
from player_recorder import Recorder

class FPS:
    def __init__(self, desired_fps=DEFAULT_FPS, invalidate_callback=None):
        self._one_second = int(1e9)
        self._frame_interval = self._one_second // desired_fps
        self._lost_time = 0
        self._frame_count = 0
        self.value = desired_fps
        self.invalidate = invalidate_callback
        self.thread = threading.Thread(target=self.loop)
        self.terminated = threading.Event()

    def run(self):
        self.thread.start()

    def loop(self):
        while not self.terminated.is_set():
            time.sleep(self.process_frame())

    def process_frame(self):
        start_time = time.perf_counter_ns()

        self.invalidate()
        self._frame_count += 1

        elapsed_time = time.perf_counter_ns() - start_time
        wait_time = self._frame_interval - elapsed_time
        if wait_time > 0:
            return wait_time*1e-9
        else:
            self._lost_time += abs(wait_time)
            return 0

    def start(self):
        self._frame_count = 0
        self._lost_time = 0
        self.start_time = time.perf_counter_ns()
        self.run()

    def stop(self):
        self.duration = time.perf_counter_ns() - self.start_time
        self.duration = self.duration/self._one_second
        self.add_lost_frames()
        self.terminated.set()

    def add_lost_frames(self):
        # calculate the expected number of frames based on the duration and frame rate
        expected_frames = self.duration * self.value
        # calculate the number of frames that have been lost
        lost_frames = self._frame_count - round(expected_frames)
        if lost_frames > 0:
            for _ in range(lost_frames+1):
                self.invalidate()

class RealTimeRecorder(Recorder):
    def __init__(self,
                 width=600,
                 height=400,
                 fps=DEFAULT_FPS,
                 video_filename='drawing.mp4',
                 invalidate_callback=None):
        super().__init__(width=width, height=height, fps=fps, video_filename=video_filename)
        self.video_filename = video_filename
        self.size = (width, height)
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.device = cv2.VideoWriter()
        self.frames = Queue(maxsize=BUFFER_SIZE)
        self.terminated = threading.Event()
        self.fps = FPS(desired_fps=fps,
                       invalidate_callback=invalidate_callback)

    def after_start(self):
        self.terminated.clear()
        self.worker_thread = threading.Thread(target=self.process_buffer)
        self.worker_thread.start()
        self.fps.start()

    def before_stop(self):
        self.fps.stop()
        self.terminated.set()
        self.frames.put('KILL')
        self.worker_thread.join()

    def put(self, frame):
        if self.device.isOpened():
            self.frames.put(frame)

    def process_buffer(self):
        while not self.terminated.is_set() or not self.frames.empty():
            data = self.frames.get()
            if self.device.isOpened() and not self.terminated.is_set():
                self.device.write(data)

    @property
    def invalidate(self):
        return self.fps.invalidate

    @invalidate.setter
    def invalidate(self, invalidate_callback):
        self.fps.invalidate = invalidate_callback

if __name__ == '__main__':
    import numpy as np

    data = np.zeros((512, 512, 3), dtype=np.uint8)

    recorder = Recorder()
    recorder.invalidate = lambda: recorder.put(data)
    recorder.start()
    time.sleep(1)
    recorder.stop()