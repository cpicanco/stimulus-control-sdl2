import os
import av
import glob
from PIL import Image
from fractions import Fraction

input_dir = os.path.dirname(os.path.abspath(__file__))
image_files = glob.glob(f"{input_dir}/*.png")
image_files_timestamps = [float(os.path.basename(i).replace(',', '.').replace('.png', '')) for i in image_files]
image_files_timestamps_dts = [t - image_files_timestamps[0] for t in image_files_timestamps]
image_files = [(x, t, d) for t, x, d in sorted(zip(image_files_timestamps, image_files, image_files_timestamps_dts))]
video_duration = image_files[-1][1]-image_files[0][1]
fps_p = len(image_files)/video_duration
fps = round(fps_p)

container = av.open("output.mp4", 'w')
video_stream = container.add_stream('libx265', rate=fps)
image = Image.open(os.path.join(input_dir, image_files[0][0]))
video_stream.width = image.width
video_stream.height = image.height
video_stream.codec_context.time_base = Fraction(1, 1000000000)

first_timestamp = None
for (image_file, timestamp, dts) in image_files:
    image_path = os.path.join(input_dir, image_file)

    frame_pts = timestamp
    frame_pts *= 1 + (fps_p - fps) / fps_p

    frame = av.VideoFrame.from_image(Image.open(image_file))
    frame.time_base = video_stream.codec_context.time_base
    frame.pts = int(round(frame_pts / video_stream.codec_context.time_base))
    frame.dts = int(round(dts / video_stream.codec_context.time_base))

    packet = video_stream.encode(frame)
    container.mux(packet)
container.mux(video_stream.encode())
container.close()
