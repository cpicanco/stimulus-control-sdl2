
import time
from queue import Queue
import threading


import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk

def get_monitor_refresh_rate():
    import win32api
    device = win32api.EnumDisplayDevices()
    settings = win32api.EnumDisplaySettings(device.DeviceName, -1)
    return int(getattr(settings,'DisplayFrequency'))

DEFAULT_FPS = int(get_monitor_refresh_rate()//3)
BUFFER_SIZE = DEFAULT_FPS*10

class FPS:
    def __init__(self, desired_fps=DEFAULT_FPS, invalidate_callback=None):
        self._one_second = 1e9
        self._frame_interval = self._one_second / desired_fps
        self._lost_time = 0
        self._frame_count = 0
        self.value = desired_fps
        self.invalidate = invalidate_callback
        self.thread = threading.Thread(target=self.loop)
        self.event = threading.Event()

    def run(self):
        self.thread.start()

    def loop(self):
        while not self.event.is_set():
            time.sleep(self.process_frame())

    def terminate(self):
        self.event.set()

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
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration = time.perf_counter() - self.start_time
        # self.duration = self.duration / 1e9
        self.add_lost_frames()

    def add_lost_frames(self):
        # calculate the expected number of frames based on the duration and frame rate
        expected_frames = self.duration * self.value
        # calculate the number of frames that have been lost
        lost_frames = self._frame_count - round(expected_frames)
        if lost_frames > 0:
            for _ in range(lost_frames+1):
                self.invalidate()


class Bitmap:
    def __init__(self, size=(512, 512)):
        self.data = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._image_id = 0
        self._image : ImageTk.PhotoImage
        self._image = None

    @property
    def size(self):
        return (self.data.shape[1], self.data.shape[0])

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def clear(self):
        self.data.fill(0)

    def new_image(self):
        self._image = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        return self._image_id, self._image

    def update(self, canvas):
        new_image = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        canvas.itemconfig(self._image_id, image=new_image)
        self._image = new_image

    @property
    def image(self):
        return self._image_id, self._image

    def setup(self, canvas):
        self._image = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        self._image_id = canvas.create_image(0, 0, anchor=tk.NW, image=self._image)

class TkinterDevice:
    def __init__(self,
                 window_name='Free Drawing',
                 bitmap_size=(512, 512),
                 video_filename='drawing.mp4',
                 fps=DEFAULT_FPS):
        self.window_name = window_name
        self.bitmap = Bitmap(bitmap_size)
        self.video_filename = video_filename
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.recorder = cv2.VideoWriter()
        self.frames = Queue(maxsize=BUFFER_SIZE)
        self.stop_thread = threading.Event()
        self.fps = FPS(desired_fps=fps,
                       invalidate_callback=self.invalidate)

        # Setup Tkinter window
        self.root = tk.Tk()
        self.root.title(window_name)
        self.canvas = Canvas(self.root,
                             bg='white',
                             width=self.bitmap.width,
                             height=self.bitmap.height)
        self.canvas.pack()
        self.bitmap.setup(self.canvas)
        self.canvas.bind("<B1-Motion>", self.draw_circle)
        self.canvas.bind("<Button-1>", self.draw_circle)

        self.canvas.bind("<Destroy>", self.terminate)

        # Setup buttons
        self.start_button = tk.Button(self.root, text='Start Recording', command=self.start_recording)
        self.start_button.pack()
        self.start_button.place(x=0, y=0)
        self.stop_button = tk.Button(self.root, text='Stop Recording', command=self.stop_recording)
        self.stop_button.pack()
        self.stop_button.place(x=100, y=0)
        self.invalidated = False

        # self.exit_button = tk.Button(self.root, text='Exit', command=self.root.quit)
        # self.exit_button.pack(side=LEFT)

    def terminate(self, event):
        self.fps.terminate()
        self.stop_recording()
        # self.root.destroy()

    def draw_circle(self, event):
        self.bitmap.clear
        x, y = event.x, event.y
        cv2.circle(self.bitmap.data, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
        self.invalidated = True

    def start_recording(self):
        if not self.recorder.isOpened():
            self.recorder.open(self.video_filename, self.fourcc, self.fps.value, self.bitmap.size)
            self.stop_thread.clear()
            self.worker_thread = threading.Thread(target=self.process_buffer)
            self.worker_thread.start()
            self.fps.start()
            print(f"Recording started, saving to '{self.video_filename}'")

    def stop_recording(self):
        if self.recorder.isOpened():
            self.fps.stop()
            self.stop_thread.set()
            self.frames.put('KILL')
            self.worker_thread.join()
            self.recorder.release()
            print(f"Recording saved with duration of ~{self.fps.duration:.2f} seconds")

    def run(self):
        # self.render()
        self.fps.run()
        self.root.mainloop()

    def invalidate(self):
        if self.invalidated:
            self.invalidated = False
            self.bitmap.update(self.canvas)

        if self.recorder.isOpened():
            self.frames.put(self.bitmap.data.copy())

    # def render(self):
    #     self.canvas.after(self.fps.process_frame(), self.render)

    def process_buffer(self):
        while not self.stop_thread.is_set() or not self.frames.empty():
            data = self.frames.get()
            if self.recorder.isOpened() and not self.stop_thread.is_set():
                self.recorder.write(data)


# Example usage
if __name__ == "__main__":
    video_device = TkinterDevice(bitmap_size=(1440, 900))
    video_device.run()