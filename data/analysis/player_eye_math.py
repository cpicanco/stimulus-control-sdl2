import numpy as np
from scipy.spatial.distance import pdist

# Screen parameters
SCREEN_W_PIX = 1440
SCREEN_H_PIX = 900

# Monitor size (in cm)
SCREEN_W_CM = 41.476  # cm
SCREEN_H_CM = 25.922  # cm
LETTER_WIDTH = 2.6  # cm
LETTER_HEIGHT = 5.3  # cm
WORD_WIDTH = 13.7  # cm
VIEWING_DISTANCE_CM = 65.0

def visual_angle(values_in_cm: float, distance_in_cm: float = VIEWING_DISTANCE_CM) -> float:
    visual_angles = 2 * np.arctan(values_in_cm / (2 * distance_in_cm))
    return np.degrees(visual_angles)

def compute_dispersion(
        points,
        viewing_distance_cm=VIEWING_DISTANCE_CM):
    if points.shape[0] < 2:
        return 0

    pairwise_distances = pdist(points, metric='euclidean')
    # Convert distances to visual angles (in radians)
    visual_angles = 2 * np.arctan(pairwise_distances / (2 * viewing_distance_cm))
    # Convert maximum visual angle to degrees
    max_dispersion = np.degrees(visual_angles.max())
    return max_dispersion

if __name__ == '__main__':
    print(f'Screen Width (Degrees): {visual_angle(SCREEN_W_CM)}')
    print(f'Word Width (Degrees): {visual_angle(WORD_WIDTH)}')
    print(f'Letter Width (Degrees): {visual_angle(LETTER_WIDTH)}')
    print('*****************************')
    print(f'Screen Height (Degrees): {visual_angle(SCREEN_H_CM)}')
    print(f'Letter Height (Degrees): {visual_angle(LETTER_HEIGHT)}')
    print('*****************************')

    print(f'Calibration Error (Degrees): {visual_angle(1.1521)}')