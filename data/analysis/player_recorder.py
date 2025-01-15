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

def record_videos():
    import os

    from fileutils import cd, data_dir
    from player_synchronizer import Synchronizer
    from player_drawings import (
        BehaviorDrawingFactory,
        GazeDrawingFactory
    )
    from explorer import load_participant_sources

    start_from_reached = False
    start_from = ''

    sources = load_participant_sources()
    for participant in sources:
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                # if 'Treino-AC-Ref-Intermitente' in source['name']:
                date = source['date']
                cycle = source['cycle']
                code = source['code']
                video_filename = code + '.mp4'
                current = f'{participant} - {date}, {cycle}, {code}'

                data_dir(verbose=False)
                cd(source['path'], verbose=False)

                # check if video already exists
                if os.path.exists(video_filename):
                    # print(f'Skipped: {current}')
                    continue

                if not start_from == '':
                    if current == start_from:
                        start_from_reached = True

                    if not start_from_reached:
                        # print(f'Skipped: {current}')
                        continue

                print(f'Recording: {current}')

                session = Synchronizer(code)

                behavior_factory = BehaviorDrawingFactory(participant)
                gaze_factory = GazeDrawingFactory(session)

                recorder = Recorder(
                    width=session.screen.width,
                    height=session.screen.height,
                    video_filename=video_filename)
                recorder.start()

                for time, frame in session.frames:
                    # if time < 30:
                    #     continue
                    # print(f'Frame: {time}')
                    behavior_data = frame[frame['label'] == 0]
                    gaze_data = frame[frame['label'] == 1]
                    img = np.zeros((session.screen.height, session.screen.width, 3), dtype=np.uint8)

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
        cv2.putText(img=frame,
                text='Quick Fox',
                org=(15, 70),
                fontHeight=60,
                color=(255,  255, 255),
                thickness=-1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=True)
        recorder.write(frame)
    recorder.stop()