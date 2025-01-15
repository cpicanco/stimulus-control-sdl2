import numpy as np

if __name__ == '__main__':
    import numpy as np
    import cv2

    def heatmap(data, detail, bins_h, bins_w, range_h, range_w):
        """ Create a blurred heatmap from gaze points. """
        xvals = data[:, 0]
        yvals = data[:, 1]

        # Create a 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            yvals, xvals,
            bins=[bins_h, bins_w],
            range=[[0, range_h], [0, range_w]]
        )

        # Apply Gaussian blur to the histogram
        kernel = detail // 2 * 2 + 1
        blurred_hist = cv2.GaussianBlur(hist, (kernel, kernel), 0)

        return blurred_hist

    # Example gaze points (x, y) with normalized values
    gaze_points = np.array([
        [0.1, 0.2],
        [0.4, 0.5],
        [0.7, 0.8],
        [0.9, 0.3],
        [0.2, 0.7],
        # Add more points as needed...
    ])

    # Canvas size (adjust as needed)
    sh, sw = 400, 400  # Height and width of the histogram
    image = np.zeros((sh, sw, 3), dtype=np.uint8)

    for pos in gaze_points:
        x = int(pos[0] * sw)  # x position in pixels
        y = int(pos[1] * sh)  # y position in pixels
        cv2.circle(image, (x, y), radius=20, color=(0, 255, 0), thickness=2)  # Green circles with radius 20


    # Detail level (kernel size for Gaussian blur)
    detail = 30

    # Generate the histogram and apply Gaussian blur
    blurred_hist = heatmap(gaze_points, detail, sh, sw, 1., 1.)

    # Normalize the result for visualization
    norm_hist = cv2.normalize(blurred_hist, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to a color map (e.g., Viridis)
    colored_hist = cv2.applyColorMap(norm_hist.astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    overlay = cv2.addWeighted(colored_hist, 1., image, 1., 0)

    # Display the result
    cv2.imshow('Gaze Heatmap', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()