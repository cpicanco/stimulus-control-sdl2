from collections import deque
import numpy as np

from player_eye_math import compute_dispersion

def fixation_from_data(gaze_data, indices):
    """
    Given the subset of gaze_data rows that form a fixation,
    return a fixation dictionary with start_time, duration, and centroid.
    """
    # start, end = base_data[0]["timestamp"], base_data[-1]["timestamp"]
    # start, end = np.searchsorted(timestamps, [start, end])
    # end = min(end, len(timestamps) - 1)  # fix `list index out of range` error
    # fix["start_frame_index"] = int(start)
    # fix["end_frame_index"] = int(end)
    # fix["mid_frame_index"] = int((start + end) // 2)

    subset = np.sort(gaze_data[indices], order='TIME_TICK')
    start_time = subset["TIME_TICK"][0]
    end_time = subset["TIME_TICK"][-1]
    duration = end_time - start_time

    fixation = {
        "GAZE_DATA": subset,
        "TIME_TICK": start_time,
        "TIME_TICK_END": end_time,
        "FPOGD": duration,
        "FPOGX": subset["BPOGX"].mean(),
        "FPOGY": subset["BPOGY"].mean(),
    }
    return fixation

class Fixation_Result_Factory:
    __slots__ = ("_id_counter",)
    def __init__(self):
        self._id_counter = 0

    def from_data(self, gaze_data, indices):
        fixation = fixation_from_data(gaze_data, indices)
        self._set_fixation_id(fixation)
        fixation_start = fixation["TIME_TICK"]
        fixation_stop = fixation_start + fixation["FPOGD"]
        return (fixation, fixation_start, fixation_stop)

    def _set_fixation_id(self, fixation):
        fixation["id"] = self._id_counter
        self._id_counter += 1

def detect_fixations(
    gaze_data, max_dispersion, min_duration, max_duration
):
    """
    Detect fixations in gaze_data DataFrame.

    Parameters
    ----------
    gaze_data : pd.DataFrame
        Must have at least: TIME_TICK, BPOGX, BPOGY.
    max_dispersion : float
        Maximum allowed dispersion (in pixels).
    min_duration : float
        Minimum duration for a fixation (same units as TIME_TICK).
    max_duration : float
        Maximum duration to consider for a fixation (same units as TIME_TICK).

    Returns
    -------
    fixations : list
        A list of tuples (fixation_dict, fixation_start, fixation_stop)
    """
    # Convert to arrays for efficiency if desired
    timestamps = gaze_data["TIME_TICK"]
    gaze = np.column_stack((gaze_data['BPOGX_D'], gaze_data['BPOGY_D']))
    fixation_result = Fixation_Result_Factory()

    working_queue = deque()  # Will store indices of rows currently considered
    remaining_indices = deque(range(len(gaze_data)))

    fixations_found = []

    while remaining_indices:
        # Check if we have at least 2 points and meet min_duration
        if len(working_queue) < 2 or (timestamps[working_queue[-1]] - timestamps[working_queue[0]]) < min_duration:
            if remaining_indices:
                idx = remaining_indices.popleft()
                working_queue.append(idx)
            continue

        # Check dispersion
        indices = np.array(working_queue)
        dispersion = compute_dispersion(gaze[indices])
        if dispersion > max_dispersion:
            # not a fixation, move forward by dropping the first element
            working_queue.popleft()
            continue
        # We found a minimal fixation
        left_idx = len(working_queue)

        # Try to extend to max_duration
        while remaining_indices:
            next_idx = remaining_indices[0]
            if timestamps[next_idx] > timestamps[working_queue[0]] + max_duration:
                break
            working_queue.append(remaining_indices.popleft())

        # Check again with extended data
        indices = np.array(working_queue)
        dispersion = compute_dispersion(gaze[indices])
        if dispersion <= max_dispersion:
            fixations_found.append(
                fixation_result.from_data(
                    gaze_data, list(working_queue))
            )
            working_queue.clear()
            continue

        # Binary search for exact end point
        slicable = list(working_queue)
        right_idx = len(working_queue)

        while left_idx < right_idx - 1:
            middle_idx = (left_idx + right_idx) // 2
            test_indices = slicable[: middle_idx + 1]
            indices = np.array(test_indices)
            dispersion = compute_dispersion(gaze[indices])
            if dispersion <= max_dispersion:
                left_idx = middle_idx
            else:
                right_idx = middle_idx

        final_base_data = slicable[:left_idx]
        to_be_placed_back = slicable[left_idx:]

        fixations_found.append(
            fixation_result.from_data(
                gaze_data, final_base_data))
        working_queue.clear()

        # Put back the data that didn't fit into the fixation
        for idx in reversed(to_be_placed_back):
            remaining_indices.appendleft(idx)

    return fixations_found

def playback(fixations):
    import cv2
    import time
    from player_eye_math import SCREEN_W_PIX, SCREEN_H_PIX

    # Create a window to display gaze points
    cv2.namedWindow('Gaze Replay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gaze Replay', SCREEN_W_PIX, SCREEN_H_PIX)

    for segment in fixations:
        gaze_dict = segment[0]
        gaze_data = gaze_dict['GAZE_DATA']
        gaze_data['BPOGX'] = gaze_data['BPOGX'] * SCREEN_W_PIX
        gaze_data['BPOGY'] = gaze_data['BPOGY'] * SCREEN_H_PIX
        gaze_dict['FPOGX'] = gaze_dict['FPOGX'] * SCREEN_W_PIX
        gaze_dict['FPOGY'] = gaze_dict['FPOGY'] * SCREEN_H_PIX

        # Playback loop for this segment
        for i in range(len(gaze_data)):
            # Create a black image for each frame (or you could persist and draw a path)
            frame = np.zeros((SCREEN_H_PIX, SCREEN_W_PIX, 3), dtype=np.uint8)
            x = int(gaze_data['BPOGX'][i])
            y = int(gaze_data['BPOGY'][i])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw fixation
            x = int(gaze_dict['FPOGX'])
            y = int(gaze_dict['FPOGY'])
            cv2.circle(frame, (x, y), 25, (0, 0, 255), 3)

            cv2.putText(frame, f"Time: {gaze_data['TIME_TICK'][i]:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('Gaze Replay', frame)

            # Calculate the delay based on time differences to simulate real-time
            if i > 0:
                dt = gaze_data['TIME_TICK'][i] - gaze_data['TIME_TICK'][i-1]
            else:
                dt = 0.02  # default small delay if first frame

            # Convert dt to milliseconds for waitKey
            # If TIME_TICK is in seconds, we can use time.sleep(dt)
            # Using waitKey alone is tricky for accurate timing. We'll do a combination.
            key = cv2.waitKey(1)
            # More accurate timing can be done with time.sleep
            # For smoother playback, handle very small dt carefully.
            time.sleep(dt)

            if key == 27:  # Press ESC to exit early
                break

    cv2.destroyAllWindows()

def playback_raw(gaze_data):
    import time

    import pandas as pd
    import cv2

    from player_eye_math import SCREEN_W_PIX, SCREEN_H_PIX

    # Create a window to display gaze points
    cv2.namedWindow('Gaze Replay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gaze Replay', SCREEN_W_PIX, SCREEN_H_PIX)

    minimun_id = gaze_data['FPOGID'].min()
    gaze_data['FPOGID'] = gaze_data['FPOGID'] - minimun_id
    gaze_data['FPOGX'] = gaze_data['FPOGX'] * SCREEN_W_PIX
    gaze_data['FPOGY'] = gaze_data['FPOGY'] * SCREEN_H_PIX

    gaze_data = pd.DataFrame(gaze_data)

    for i, fixation in gaze_data.groupby('FPOGID'):
        fx = fixation[fixation['FPOGV'] == 1]
        x_mean = fx['FPOGX'].mean()
        y_mean = fx['FPOGY'].mean()

        if pd.isna(x_mean) or pd.isna(y_mean):
            continue

        x_mean = int(x_mean)
        y_mean = int(y_mean)
        for index, row in fx.iterrows():
            frame = np.zeros((SCREEN_H_PIX, SCREEN_W_PIX, 3), dtype=np.uint8)
            x, y = int(row['FPOGX']), int(row['FPOGY'])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (x_mean, y_mean), 25, (0, 255, 0), 3)

            cv2.putText(frame, f"{row['FPOGID']}", (x_mean+20, y_mean-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(frame, f"Duration: {row['FPOGD']:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow('Gaze Replay', frame)

            # Calculate the delay based on time differences to simulate real-time
            if i > 0:
                dt = gaze_data['TIME_TICK'][i] - gaze_data['TIME_TICK'][i-1]
            else:
                dt = 0.02

            key = cv2.waitKey(1)
            time.sleep(dt)

            if key == 27:  # Press ESC to exit early
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    import os

    from fileutils import cd, data_dir
    from explorer import load_participants
    from player_information import GazeInfo
    from player_synchronizer import Synchronizer
    from numpy.lib import recfunctions as rfn

    sources = load_participants()
    for participant in sources:
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                # if 'Treino-AC-Ref-Intermitente' in source['name']:
                date = source['date']
                cycle = source['cycle']
                code = source['code']
                print(code)
                if code == '004':
                    gaze_data_filename = code + '.gaze'
                    current = f'{participant} - {date}, {cycle}, {code}'

                    data_dir(verbose=False)
                    cd(os.path.join(participant, 'analysis', date, cycle), verbose=False)

                    info = GazeInfo(code)
                    session = Synchronizer(info)
                    # exit all for loops
                    break
                else:
                    continue
            if code == '004':
                break
            else:
                continue
        if code == '004':
            break
        else:
            continue

    # Filter only valid gaze points (BPOGV == 1)
    print(f'Loading gaze data from {current}')

    playback_raw(session.gaze.events)
    # denorm_x, denorm_y = session.gaze.denormalize(measurement='fixations')
    # valid_mask = session.gaze.events["BPOGV"] == 1
    # valid_data = session.gaze.events[valid_mask]
    # gaze_data = valid_data[["TIME_TICK", "BPOGX", "BPOGY"]].copy()
    # # add denormalized as new columns
    # gaze_data = rfn.append_fields(gaze_data, 'BPOGX_D', data=denorm_x, usemask=False)
    # gaze_data = rfn.append_fields(gaze_data, 'BPOGY_D', data=denorm_y, usemask=False)

    # if len(gaze_data) == 0:
    #     print("No valid gaze data")
    #     exit()

    # MAX_DISPERSION_DEG = 1.50  # degrees of visual angle
    # MIN_DURATION_MS = 0.100     # minimum duration in s
    # MAX_DURATION_MS = 0.220     # maximum duration in s

    # fixations_found = detect_fixations(
    #     gaze_data=gaze_data,
    #     max_dispersion=MAX_DISPERSION_DEG,
    #     min_duration=MIN_DURATION_MS,
    #     max_duration=MAX_DURATION_MS
    # )

    # print(f'Found {len(fixations_found)} fixations.')

    # # Run the playback
    # playback(fixations_found)