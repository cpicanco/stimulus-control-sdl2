import itertools
from collections import OrderedDict

from matplotlib.lines import Line2D
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from fileutils import data_dir, cd
from player_media import draw_word, load_fonts
from player_information import GazeInfo
from player_synchronizer import Synchronizer

load_fonts()

def empty_word(width, height):
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    word_image = draw_word(buffer, 0, 0, '')

    # draw a white rectangle of 200 pixels height
    rect_width = 100
    spacing = (width // 4)
    for i in range(4):
        x = 15 + (i * spacing)  # Starting at x = 10, then spaced
        y = 15  # Constant y position
        cv2.rectangle(
            word_image,
            (x, y),
            (x + rect_width, y + 200),
            (255, 255, 255),
            3,
            cv2.LINE_AA)
    return word_image


# Create a histogram with clamped x-values
def word_heatmap(x_data, y_data, weights, word_info, title):
    word_path = word_info['path']
    word_text = word_info['word']
    rectangle = word_info['rectangle']

    width = rectangle.width
    height = rectangle.height
    buffer = np.zeros((height, width, 3), dtype=np.uint8)
    if word_text is None:
        word_image = empty_word(width, height)
    else:
        word_image = draw_word(buffer, 0, 0, word_text)

    aspect_ratio = height / width
    horizontal_bins = 4
    vertical_bins = 2

    # Create the histogram plot
    plt.figure(figsize=(5, 2.8))
    ax = plt.gca()
    heatmap_data, xedges, yedges = np.histogram2d(
        x_data, y_data,
        bins=[horizontal_bins, vertical_bins],
        range=[[0, width], [0, height]],
        weights=weights)

    plt.xlabel(f'Posição horizontal do olhar\n({horizontal_bins} grupos de {width//horizontal_bins} pixels)')
    plt.ylabel(f'Posição vertical do olhar\n({vertical_bins} grupos de {height//vertical_bins} pixels)')
    plt.title(title)

    # convert image to grayscale
    gray_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

    # extract word contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    image_bgr = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize RGBA array
    # draw contours
    cv2.drawContours(image_bgr, contours, -1, (255, 255, 255), 2)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to RGBA
    image_rgba = np.zeros((height, width, 4), dtype=np.uint8)  # Initialize RGBA array
    image_rgba[..., :3] = image_rgb  # Copy the RGB channels
    image_rgba[..., 3] = 255  # Set full opacity initially (alpha = 255)

    # Make the black pixels transparent
    black_pixels = np.all(image_rgb == [0, 0, 0], axis=-1)  # Find black pixels
    image_rgba[black_pixels, 3] = 0  # Set their alpha channel to 0 (transparent)

    plt.imshow(
        heatmap_data.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='upper',
        cmap='viridis')

    color_bar = plt.colorbar(label='Duração das fixações (segundos)')
    # Get the minimum and maximum values from the data
    min_val, max_val = heatmap_data.min(), heatmap_data.max()
    mid_val = (max_val) / 2

    # Set the ticks of the color bar to include min and max values
    color_bar.set_ticks([min_val, mid_val, max_val])
    color_bar.set_ticklabels([f'{min_val:.2f}', f'{mid_val:.2f}', f'{max_val:.2f}'])

    plt.imshow(
        image_rgba,
        extent=[0, width, 0, height],
        aspect=aspect_ratio,
        alpha=0.5)

    ax.set_aspect(aspect='auto')
    plt.tight_layout()

    # Show the final plot with the overlay
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'word_overlay_{word_path}.pdf')
    plt.close()
    cd('..')
    cd('..')

def draw_individual_heatmaps(sources):
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        info = GazeInfo(code)
        session = Synchronizer(info)

        for word_text in [source['word1'], source['word2']]:
            gaze_relative_to_words = session.word_filter(word_text)
            title = f'Fixações do olhar\nsobre a palavra {word_text} como S+'

            word_dict = {
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}',
                'rectangle': None,
                'word': word_text
            }
            filtered_data = gaze_relative_to_words.by_positive
            fixations = filtered_data.fixations
            word_dict['rectangle'] = filtered_data.rectangle

            word_heatmap(
                fixations['FPOGX'].values,
                fixations['FPOGY'].values,
                fixations['FPOGD'].values,
                word_dict,
                title)


            gaze_relative_to_words = session.word_filter(word_text)
            title = f'Fixações do olhar\nsobre palavras S-, {word_text} como S+'
            word_dict = {
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'rectangle': None,
                'word': None
            }

            filtered_data = gaze_relative_to_words.by_negative
            fixations = filtered_data.fixations
            word_dict['rectangle'] = filtered_data.rectangle

            word_heatmap(
                fixations['FPOGX'].values,
                fixations['FPOGY'].values,
                fixations['FPOGD'].values,
                word_dict,
                title)

def get_heatmap_data(sources):
    data = []
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        session = Synchronizer(GazeInfo(code))
        words = []
        for word_text in source['words']:
            session_filtered = session.word_filter(word_text)
            session_filtered.process_trials_with_reference_word()

            heatmap_data = session_filtered.get_heatmap_of_positive_words()
            positive = {
                'word': word_text,
                'word_type': 'positive',
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}',
                'heatmap': heatmap_data['heatmap'],
                'heatmap_xmesh': heatmap_data['heatmap_xmesh'],
                'heatmap_ymesh': heatmap_data['heatmap_ymesh'],
                'rectangle': heatmap_data['rectangle'],
                'title': f'{word_text}'
            }

            heatmap_data = session_filtered.get_heatmap_of_negative_words()
            negative = {
                'word': word_text,
                'word_type': 'negative',
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'heatmap': heatmap_data['heatmap'],
                'heatmap_xmesh': heatmap_data['heatmap_xmesh'],
                'heatmap_ymesh': heatmap_data['heatmap_ymesh'],
                'rectangle': heatmap_data['rectangle'],
                'title': f'{word_text}'
            }

            words.append(positive)
            words.append(negative)

        data.append(words)

    return data

# def get_heatmap_data_single(sources):
#     from player_information import GazeInfo
#     from player_synchronizer import Synchronizer

#     data = []
#     for source in sources:
#         participant = source['participant']
#         date = source['date']
#         cycle = source['cycle']
#         code = source['code']
#         data_dir()
#         cd(participant)
#         cd('analysis')
#         cd(date)
#         cd(cycle)

#         session = Synchronizer(GazeInfo(code))
#         words = []
#         for word_text in source['words']:
#             session_filtered = session.word_filter(word_text)
#             filtered_gaze = session_filtered.
#             heatmap_data = filtered_gaze.get_heatmap()
#             all = {
#                 'heatmap': heatmap_data['heatmap'],
#                 'heatmap_xmesh': heatmap_data['heatmap_xmesh'],
#                 'heatmap_ymesh': heatmap_data['heatmap_ymesh'],
#             }

#             words.append(all)

#         data.append(words)

#     return data

def draw_uniform_heatmap(ax: axes.Axes, data: dict):
    # word_text = data['word']
    # rectangle = data['rectangle']
    heatmap_data = data['heatmap']
    xedges = data['heatmap_xmesh'][0]
    yedges = data['heatmap_ymesh'].T[0]

    # width = rectangle.width
    # height = rectangle.height
    # buffer = np.zeros((height, width, 3), dtype=np.uint8)
    # if word_text is None:
    #     word_image = empty_word(width, height)
    # else:
    #     word_image = draw_word(buffer, 0, 0, word_text)

    # aspect_ratio = height / width

    # # convert image to grayscale
    # gray_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

    # # extract word contours
    # contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # image_bgr = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize RGBA array
    # # draw contours
    # cv2.drawContours(image_bgr, contours, -1, (255, 255, 255), 2)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # # Convert the RGB image to RGBA
    # image_rgba = np.zeros((height, width, 4), dtype=np.uint8)  # Initialize RGBA array
    # image_rgba[..., :3] = image_rgb  # Copy the RGB channels
    # image_rgba[..., 3] = 255  # Set full opacity initially (alpha = 255)

    # # Make the black pixels transparent
    # black_pixels = np.all(image_rgb == [0, 0, 0], axis=-1)  # Find black pixels
    # image_rgba[black_pixels, 3] = 0  # Set their alpha channel to 0 (transparent)

    mappable = ax.imshow(
        # heatmap_data.T[[1, 0], :],
        heatmap_data.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmin=0,
        # vmax=overall_max
        cmap='viridis')

    color_bar = plt.colorbar(
        mappable,
        ax=ax)
    # Get the minimum and maximum values from the data
    # min_val, max_val = heatmap_data.min(), heatmap_data.max()
    max_val = heatmap_data.max()
    mid_val = (max_val) / 2

    # Set the ticks of the color bar to include min and max values
    color_bar.set_ticks([0, mid_val, max_val])
    color_bar.set_ticklabels([f'{0:.2f}', f'{mid_val:.2f}', f'{max_val:.2f}'])

    # ax.imshow(
    #     image_rgba,
    #     extent=[0, width, 0, height],
    #     aspect=aspect_ratio,
    #     alpha=0.5)

    ax.set_aspect(aspect='auto')


def draw_uniform_heatmaps(sources: list):
    # Sample data for heatmap (you can use your own data instead)
    data = get_heatmap_data(sources)
    # Create a 4x6 grid of subplots for the heatmaps
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(8, 6), sharex=True, sharey=True)
    axes[5, 1].set_xlabel('Horizontal gaze position')  # Bottom row, second column
    axes[2, 0].set_ylabel('Vertical gaze position')    # Middle row, first column
    axes[0, 0].set_title('S+')
    axes[0, 1].set_title('S-')
    axes[0, 2].set_title('S+')
    axes[0, 3].set_title('S-')
    # Iterating over each subplot and adding a heatmap using draw_heatmap function
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            draw_uniform_heatmap(axes[row, col], data[row][col])  # Call the function for each subplot

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Add vertical text outside the axes
    for row in range(6):
        # Get the position of the first axes in the row
        pos_left = axes[row][0].get_position()
        # Calculate the y-center of the axes
        top = pos_left.y0 + pos_left.height + 0.015

        # Add text to the left
        fig.text(
            pos_left.x0,          # Slightly to the left of the first axes
            top,                    # Vertically centered
            data[row][0]['title'],        # Text from your data
            ha='left',                  # Horizontal alignment to the right
            va='center',                 # Vertical alignment centered
            # rotation='vertical',         # Rotate text vertically
            fontsize=11
        )

        pos_left = axes[row][2].get_position()
        # Add text to the right
        fig.text(
            pos_left.x0,         # Slightly to the right of the last axes
            top,              # Vertically centered
            data[row][2]['title'],        # Text from your data
            ha='left',                   # Horizontal alignment to the left
            va='center',                 # Vertical alignment centered
            # rotation=-90,         # Rotate text vertically
            fontsize=11
        )

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'words_overlay_uniform_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

def get_midpoint(ax_left, ax_right):
    """
    Calculate the midpoint between two Axes objects horizontally.
    Returns the x-coordinate in figure coordinates.
    """
    pos_left = ax_left.get_position()
    pos_right = ax_right.get_position()
    # Midpoint is the average of the right edge of the left Axes and the left edge of the right Axes
    midpoint = (pos_left.x0 + pos_right.x0) / 2
    return midpoint

def draw_non_uniform_heatmaps(sources: dict):
    def calculate_overall_min_max(data):
        overall_min = 0
        overall_max = 0
        for row, _ in enumerate(data):
            for col, _ in enumerate(data[row]):
                overall_min = min(overall_min, data[row][col]['heatmap'].min())
                overall_max = max(overall_max, data[row][col]['heatmap'].max())

        for row, _ in enumerate(data):
            for col, _ in enumerate(data[row]):
                data[row][col]['heatmap_overall_min'] = overall_min
                data[row][col]['heatmap_overall_max'] = overall_max

        return overall_min, overall_max

    def draw_non_uniform_heatmap(ax: axes.Axes, data: dict):
        heatmap_data = data['heatmap']
        X = data['heatmap_xmesh']
        Y = data['heatmap_ymesh']
        overall_min = data['heatmap_overall_min']
        overall_max = data['heatmap_overall_max']

        data['heatmap_mesh'] = ax.pcolormesh(
            X, Y, heatmap_data.T,
            shading='flat',
            cmap='viridis',
            vmin=overall_min,
            vmax=overall_max
        )

    def draw_colorbar(fig, heatmap_mesh, axs, min_value, max_value):
        color_bar = fig.colorbar(heatmap_mesh, ax=axs.ravel().tolist())

        mid_value = (max_value) / 2

        # Set the ticks of the color bar to include min, max, and mid values
        color_bar.set_ticks([min_value, mid_value, max_value])
        color_bar.set_ticklabels([f'{min_value:.2f}', f'{mid_value:.2f}', f'{max_value:.2f}'])

    def draw_word_overlay(ax: axes.Axes, data: dict):
        word_text = data['word']
        rectangle = data['rectangle']
        width = rectangle.width
        height = rectangle.height

        buffer = np.zeros((height, width, 3), dtype=np.uint8)
        if word_text is None:
            word_image = empty_word(width, height)
        else:
            word_image = draw_word(buffer, 0, 0, word_text)

        aspect_ratio = height / width

        # convert image to grayscale
        gray_image = cv2.cvtColor(word_image, cv2.COLOR_BGR2GRAY)

        # extract word contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_bgr = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize RGBA array
        # draw contours
        cv2.drawContours(image_bgr, contours, -1, (255, 255, 255), 2)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to RGBA
        image_rgba = np.zeros((height, width, 4), dtype=np.uint8)  # Initialize RGBA array
        image_rgba[..., :3] = image_rgb  # Copy the RGB channels
        image_rgba[..., 3] = 255  # Set full opacity initially (alpha = 255)

        # Make the black pixels transparent
        black_pixels = np.all(image_rgb == [0, 0, 0], axis=-1)  # Find black pixels
        image_rgba[black_pixels, 3] = 0  # Set their alpha channel to 0 (transparent)

        ax.imshow(
            image_rgba,
            extent=[0, width, 0, height],
            aspect=aspect_ratio,
            alpha=0.5)

    data = get_heatmap_data(sources)
    # here a single axis is used for all heatmaps, so that the colorbar is shared
    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(8, 6), sharex=True, sharey=True)

    axs[0, 0].set_title('S+')
    axs[0, 1].set_title('S-')
    axs[0, 2].set_title('S+')
    axs[0, 3].set_title('S-')
    min_value, max_value = calculate_overall_min_max(data)

    # draw heatmaps first
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            draw_non_uniform_heatmap(axs[row][col], data[row][col])  # Call the function for each subplot


    plt.tight_layout(rect=[0.05, 0, 0.95, 1])  # Adjust left and right margins

    mesh = data[row][col]['heatmap_mesh']

    # draw a single colorbar for all heatmaps BEFORE drawing word overlays
    draw_colorbar(fig, mesh, axs, min_value, max_value)


    # Add vertical text outside the axes
    for row in range(6):
        # Get the position of the first axes in the row
        pos_left = axs[row][0].get_position()
        # Calculate the y-center of the axes
        top = pos_left.y0 + pos_left.height + 0.015

        # Add text to the left
        fig.text(
            pos_left.x0,          # Slightly to the left of the first axes
            top,                    # Vertically centered
            data[row][0]['title'],        # Text from your data
            ha='left',                  # Horizontal alignment to the right
            va='center',                 # Vertical alignment centered
            # rotation='vertical',         # Rotate text vertically
            fontsize=12
        )

        pos_left = axs[row][2].get_position()
        top = pos_left.y0 + pos_left.height + 0.015
        # Add text to the right
        fig.text(
            pos_left.x0,         # Slightly to the right of the last axes
            top,              # Vertically centered
            data[row][2]['title'],        # Text from your data
            ha='left',                   # Horizontal alignment to the left
            va='center',                 # Vertical alignment centered
            # rotation=-90,         # Rotate text vertically
            fontsize=12
        )

    # Adjust layout to make space for the vertical texts

    # Optional: Add overall figure title or axis labels
    # fig.suptitle('Your Figure Title', fontsize=16)
    fig.text(0.5, 0.005, 'Horizontal gaze position', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Vertical gaze position', rotation='vertical', fontsize=12)

    # draw participant name at top right
    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    # word overlays must be drawn after heatmaps to avoid wrong values in colorbar
    # for row, _ in enumerate(data):
    #     for col, _ in enumerate(data[row]):
    #         draw_word_overlay(axs[row][col], data[row][col])

    # plt.tight_layout()

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'words_overlay_non_uniform_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

def get_switching_data(sources: list):
    data = []
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        session = Synchronizer(GazeInfo(code))
        switchings = []
        for word_text in source['words']:
            session_filtered = session.word_filter(word_text)
            session_filtered.process_trials_with_reference_word()
            switching = {
                'word': word_text,
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'switching_rate': session_filtered.switching_rate
            }

            switchings.append(switching)

        data.append(switchings)

    return data

def draw_switchings(sources: list):
    def calculate_maximum_switchings(data):
        max_value = 0
        for row, _ in enumerate(data):
            for col, _ in enumerate(data[row]):
                max_value = max(max_value, max(data[row][col]['switching_rate']))
        return max_value
    data = get_switching_data(sources)
    # here a single axis is used for all heatmaps, so that the colorbar is shared
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    axs[5, 1].set_xlabel('Trials')  # Bottom row, second column
    axs[2, 0].set_ylabel('Switchings')    # Middle row, first column

    # draw heatmaps first
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            # plot lines
            y_data = data[row][col]['switching_rate']
            x_data = [i+1 for i in range(len(y_data))]
            axs[row][col].plot(
                x_data,
                y_data,
                color='black',
                linewidth=1)
            # text on top left
            axs[row][col].text(
                0.01,
                0.99,
                f'{data[row][col]["word"]}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=axs[row][col].transAxes)

    max_swichings = calculate_maximum_switchings(data)

    plt.ylim(-0.2, max_swichings + 0.2)
    plt.yticks(np.arange(0, max_swichings + 1, 2))


    plt.tight_layout()

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'switchings_per_second_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

def get_trial_duration_data(sources: list):
    data = []
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        session = Synchronizer(GazeInfo(code))
        trial_duration = []
        for word_text in source['words']:
            session_filtered = session.word_filter(word_text)
            session_filtered.process_trials_with_reference_word()
            formated_data = {
                'word': word_text,
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'trial_duration': session_filtered.duration_per_trial
            }

            trial_duration.append(formated_data)

        data.append(trial_duration)

    return data

def draw_trial_duration(sources: list):
    def calculate_maximum_duration(data):
        max_duration = 0
        for row, _ in enumerate(data):
            for col, _ in enumerate(data[row]):
                max_duration = max(max_duration, max(data[row][col]['trial_duration']))
        return max_duration

    data = get_trial_duration_data(sources)
    # here a single axis is used for all heatmaps, so that the colorbar is shared
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    axs[5, 1].set_xlabel('Trials')  # Bottom row, second column
    axs[2, 0].set_ylabel('Duration (seconds)')    # Middle row, first column

    max_duration = calculate_maximum_duration(data)

    # draw heatmaps first
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            # plot lines
            y_data = data[row][col]['trial_duration']
            x_data = [i+1 for i in range(len(y_data))]
            axs[row][col].plot(
                x_data,
                y_data,
                color='black',
                linewidth=1)
            # text on top left
            axs[row][col].text(
                0.01,
                0.99,
                f'{data[row][col]["word"]}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=axs[row][col].transAxes)

    plt.ylim(-0.2, max_duration+0.2)
    # plt.yticks([0, 2, 4, 6])


    plt.tight_layout()

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'trial_duration_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

def get_fixations_data(sources: list):
    data = []
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        session = Synchronizer(GazeInfo(code))
        switchings = []
        for word_text in source['words']:
            session_filtered = session.word_filter(word_text)
            session_filtered.process_trials_with_reference_word()
            switching = {
                'word': word_text,
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'fixations': session_filtered.switching_rate
            }

            switchings.append(switching)

        data.append(switchings)

    return data

def draw_fixations(sources: list):
    data = get_fixations_data(sources)
    # here a single axis is used for all heatmaps, so that the colorbar is shared
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    axs[5, 1].set_xlabel('Seconds (groups of 5)')  # Bottom row, second column
    axs[2, 0].set_ylabel('Fixations')    # Middle row, first column

    # draw heatmaps first
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            # axs[row][col]
            # data[row][col]
            # plot lines
            axs[row][col].plot(
                data[row][col]['fixations'],
                color='black',
                linewidth=1)

    # plt.ylim(-0.5, 6.5)
    # plt.yticks([0, 2, 4, 6])

    plt.tight_layout()

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'fixations_normalized{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

def draw_y_velocity(sources: list):
    def get_velocity_data(sources: list):
        data = []
        for source in sources:
            participant = source['participant']
            date = source['date']
            cycle = source['cycle']
            code = source['code']
            data_dir()
            cd(participant)
            cd('analysis')
            cd(date)
            cd(cycle)

            session = Synchronizer(GazeInfo(code))
            gaze = session.gaze.DataFrame
            gaze = gaze[gaze['FPOGV'] == 1]
            # gaze = gaze[gaze['TIME_TICK'] < 10]
            gaze['delta_y'] = gaze['FPOGY'].diff()
            gaze['delta_t'] = gaze['TIME_TICK'].diff()
            gaze['y_velocity'] = gaze['delta_y'] / gaze['delta_t']

            # Calculate delta_velocity and delta_t for acceleration
            gaze['delta_velocity'] = gaze['y_velocity'].diff()
            gaze['delta_t_acc'] = gaze['TIME_TICK'].diff()

            # Calculate y_acceleration
            gaze['y_acceleration'] = gaze['delta_velocity'] / gaze['delta_t_acc']


            velocity = {
                'time': gaze['TIME_TICK'],
                'y_acceleration': gaze['y_acceleration'],
                'y_velocity': gaze['y_velocity'],
                'path': f'{participant}_cycle_{cycle}_session_{code}_y_velocity',
            }
            data.append(velocity)

        return data
    data = get_velocity_data(sources)
    # here a single axis is used for all heatmaps, so that the colorbar is shared
    fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(8, 6), sharex=True, sharey=True)
    axs[5].set_xlabel('Seconds')  # Bottom row, second column
    axs[2].set_ylabel('y velocity')    # Middle row, first column

    # draw heatmaps first
    for row, _ in enumerate(data):
        axs[row].plot(
            data[row]['time'],
            data[row]['y_velocity'],
            color='black',
            linewidth=1)
        axs[row].plot(
            data[row]['time'],
            data[row]['y_acceleration'],
            color='orange',
            linewidth=0.5)

    plt.tight_layout()

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'y_velocity_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')

# def draw_heatmaps_CD1(sources: list):
#     data = get_heatmap_data(sources)


def get_heatmap_by_trial(sources):
    def concatenate_heatmaps(heatmap_by_trial: list):
        # get the first heatmap
        heatmap = heatmap_by_trial[0]
        # loop through the rest of the heatmaps and concatenate them to the first heatmap
        for i in range(1, len(heatmap_by_trial)):
            heatmap = np.concatenate((heatmap, heatmap_by_trial[i]), axis=1)
        return heatmap

    data = []
    heatmaps_maximum = 0
    for source in sources:
        participant = source['participant']
        date = source['date']
        cycle = source['cycle']
        code = source['code']
        data_dir()
        cd(participant)
        cd('analysis')
        cd(date)
        cd(cycle)

        session = Synchronizer(GazeInfo(code))
        words = []
        for word_text in source['words']:
            session_filtered = session.word_filter(word_text)
            session_filtered.process_trials_with_reference_word()

            heatmap_data = session_filtered.get_heatmap_of_positive_words()
            positive = {
                'word': word_text,
                'word_type': 'positive',
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}',
                'heatmap_by_trial': concatenate_heatmaps(heatmap_data['heatmap_by_trial']),
                'title': f'{word_text}'
            }

            heatmap_data = session_filtered.get_heatmap_of_negative_words()
            negative = {
                'word': word_text,
                'word_type': 'negative',
                'path': f'{participant}_cycle_{cycle}_session_{code}_word_{word_text}_negative',
                'heatmap_by_trial': concatenate_heatmaps(heatmap_data['heatmap_by_trial']),
                'title': f'{word_text}'
            }
            heatmaps_maximum = max(heatmaps_maximum, positive['heatmap_by_trial'].max(), negative['heatmap_by_trial'].max())
            words.append(positive)
            words.append(negative)

        data.append(words)

    for source in data:
        for word in source:
            word['heatmaps_maximum'] = heatmaps_maximum

    return data

def draw_heatmap_by_trial(ax: axes.Axes, data: dict):
    heatmap_data = data['heatmap_by_trial']
    heatmaps_max = data['heatmaps_maximum']

    mappable = ax.imshow(
        heatmap_data.T,
        extent=[0, 500, heatmap_data.shape[1], 0],
        vmin=0,
        vmax=heatmaps_max,
        # origin='upper',
        cmap='viridis')

    color_bar = plt.colorbar(
        mappable,
        ax=ax)


    # change font size of y ticks
    # ax.tick_params(axis='y', labelsize=8)

    # y ticks and labels
    ax.set_yticks(np.arange(heatmap_data.shape[1])+0.5)
    ax.set_yticklabels(np.arange(1, heatmap_data.shape[1]+1))


    # Get the minimum and maximum values from the data
    # min_val, max_val = heatmap_data.min(), heatmap_data.max()
    max_val = heatmap_data.max()
    mid_val = (max_val) / 2

    # Set the ticks of the color bar to include min and max values
    color_bar.set_ticks([0, mid_val, max_val])
    color_bar.set_ticklabels([f'{0:.2f}', f'{mid_val:.2f}', f'{max_val:.2f}'])

    ax.set_aspect(aspect='auto')


def draw_heatmaps_by_trial(sources: list):
    # Sample data for heatmap (you can use your own data instead)
    data = get_heatmap_by_trial(sources)
    # Create a 4x6 grid of subplots for the heatmaps
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(6, 8), sharex=True, sharey=True)
    axes[5, 1].set_xlabel('Horizontal gaze position')  # Bottom row, second column
    axes[2, 0].set_ylabel('Trials')    # Middle row, first column
    axes[0, 0].set_title('S+', pad=20)
    axes[0, 1].set_title('S-', pad=20)
    axes[0, 2].set_title('S+', pad=20)
    axes[0, 3].set_title('S-', pad=20)
    # Iterating over each subplot and adding a heatmap using draw_heatmap function
    for row, _ in enumerate(data):
        for col, _ in enumerate(data[row]):
            draw_heatmap_by_trial(axes[row, col], data[row][col])  # Call the function for each subplot

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Add vertical text outside the axes
    for row in range(6):
        # Get the position of the first axes in the row
        pos_left = axes[row][0].get_position()
        # Calculate the y-center of the axes
        top = pos_left.y0 + pos_left.height + 0.015

        # Add text to the left
        fig.text(
            pos_left.x0,             # Slightly to the left of the first axes
            top,                     # Vertically centered
            data[row][0]['title'],   # Text from your data
            ha='left',               # Horizontal alignment to the right
            va='center',             # Vertical alignment centered
            # rotation='vertical',   # Rotate text vertically
            fontsize=11
        )

        pos_left = axes[row][2].get_position()
        # Add text to the right
        fig.text(
            pos_left.x0,                # Slightly to the right of the last axes
            top,                        # Vertically centered
            data[row][2]['title'],      # Text from your data
            ha='left',                  # Horizontal alignment to the left
            va='center',                # Vertical alignment centered
            # rotation=-90,             # Rotate text
            fontsize=11
        )

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'heatmap_by_trial_{sources[0]["participant"]}.pdf')
    plt.close()
    cd('..')
    cd('..')



def calculate(sources: list):
    uid = 1
    for source in sources:
        session : Synchronizer
        session = source['session']

        session_filtered = session.word_filter('')
        trials = session_filtered.trials
        source['data'] = {}
        for relation in session.info.relations:
            source['data'][relation] = [t for t in trials if t['relation'] == relation]
            for d in source['data'][relation]:
                d['relation'] = relation
                d['uid'] = uid
                uid += 1

def calculate_rolling(df, window=4, result_column='result', percentage=True):
    """
    Groups the DataFrame by specified group_size and calculates the percentage of target in result_column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - window (int): The number of rows per group.
    - result_column (str): The column to evaluate (default is 'Result').
    - percentage (bool): Whether to calculate the percentage (default is True).

    Returns:
    - pd.DataFrame: A DataFrame with group numbers and their corresponding hit percentages.
    """

    if percentage:
        percent = 100
    else:
        percent = 1

    df = df.copy()
    df['rolling_percentage'] = df[result_column].rolling(window=window).mean() * percent

    # Group data into chunks of size `window`
    # Apply the mean calculation and scale
    df['rolling_percentage_no_overlap'] = df[result_column].groupby(df.index // window).transform('mean') * percent

    return df

def pairwise_non_overlapping(iterable):
    it = iter(iterable)
    return list(zip(it, it))


marker1 = 'x'
marker2 = 'x_'
marker3 = 'o'
marker4 = 'o_'

def hits_from_sources(sources: list):
    result = []
    for source in sources:
        data = []
        for relation in source['data']:
            for d in source['data'][relation]:
                data.append(d)
        result.append(calculate_rolling(pd.DataFrame(data)))

    df = pd.concat(result, axis=0, ignore_index=True)

    color_map = {
        '1': 'gray',
        '2a': 'gray',
        '2b': 'gray',
        '3': 'black',
        '4': 'black',
        '5': 'black',
        '6': 'black',
        '7': 'black'
    }
    df['edgecolors'] = df['condition'].map(color_map)

    facecolor_map = {
        '1': 'gray',
        '2a': 'gray',
        '2b': 'gray',
        '3': 'black',
        '4': 'none',
        '5': 'none',
        '6': 'none',
        '7': 'black'
    }
    df['facecolors'] = df['condition'].map(facecolor_map)

    marker_map = {
        '1': marker1,
        '2a': marker1,
        '2b': marker1,
        '3': marker2,
        '4': marker3,
        '5': marker3,
        '6': marker3,
        '7': marker4
    }
    df['marker'] = df['condition'].map(marker_map)

    # normalize session column
    df['file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
    unique_sorted = df['file_num'].sort_values().unique()
    mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
    df['session'] = df['file_num'].map(mapping)
    df = df.drop(['file_num', 'file'], axis=1)

    return df

def hits_by_trial_single(data: pd.DataFrame, sources: list, filter=None):
    def draw_hits(ax: plt.Axes, group: pd.DataFrame, vertical_lines: list):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # draw horizontal line for 0%, 50% of trials, 100%
        for y in [0, 50, 100]:
            ax.axhline(y=y, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

        marker_label_map = {
            marker1: 'teaching',
            marker2: 'assessment (teaching words)',
            marker3: 'assessment (generalization words)',
            marker4: 'assessment (MPP)'
        }

        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['rolling_percentage_no_overlap'],
                marker=marker.replace('_', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        ax.set_ylim([-20, 120])
        ax.set_yticks([0, 50, 100])

        relation = group['relation'].unique()[0]

        mapping = {
            'AB' : 'listen to word then select picture',
            'AC' : 'listen to word then select written word',
            'BC' : 'observe picture then select written word',
            'CB' : 'observe written word then select picture',
            'CD' : 'oral reading/textual behavior'
        }
        # Get the position of the axes in figure coordinates
        ax.text(-15, 150, f'{relation} - {mapping[relation]}', ha='left', va='center', fontsize=9)

    if filter is not None:
        data = data[data['relation'].isin(filter)]

    number_of_rows =len(data['relation'].unique())
    unique_sessions = data['session'].unique()

    if number_of_rows == 1:
        figsize = (5, 2)
    else:
        figsize = (8, 5)
        cycles_onset = []
        vertical_lines = []
        for __, g in data.groupby(['cycle', 'condition', 'session']):
            if g['condition'].unique()[0] == '7':
                vertical_lines.append(g['uid'].values[0])
                vertical_lines.append(g['uid'].values[-1])
                if g['session'].unique()[0] == unique_sessions[-1]:
                    continue
                else:
                    cycles_onset.append(g['uid'].values[-1])

            # if g['condition'].unique()[0] == '2b':
            #     vertical_lines.append(g['uid'].values[-1])

        # remove first and last vertical line
        vertical_lines.sort()
        vertical_lines = vertical_lines[1:-1]

    fig, ax = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=figsize,
        sharex=True, sharey=True)

    if number_of_rows == 1:
        ax = [ax]

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        if number_of_rows == 1:
            # reset uid
            group['uid'] = range(1, len(group) + 1)
            vertical_lines = []
            cycles_onset = []
            if relation ==  'CD':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '7':
                        vertical_lines.append(g['uid'].values[0])
                        if filter[0] in ['CD', 'AB']:
                            vertical_lines.append(g['uid'].values[-1])

                        if g['session'].unique()[0] == unique_sessions[-1]:
                            continue
                        else:
                            cycles_onset.append(g['uid'].values[-1])

                cycles_onset.append(group['uid'].max())
                ax[idx].set_xticks(cycles_onset)
                # remove first and last vertical line
                vertical_lines.sort()
                vertical_lines = vertical_lines[1:-1]

                for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

            if relation == 'AB':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '1':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())

                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

            if relation ==  'AC':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '2a':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())

                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

            if (relation ==  'BC') or (relation == 'CB'):
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '3':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())
                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

        draw_hits(ax[idx], group, vertical_lines)

    if number_of_rows == 1:
        ax[0].set_xlabel('Trials (groups of 4, with overlapping)')
        ax[0].set_ylabel('Percent correct\n(rolling average)')
    else:
        ax[2].set_ylabel('Percent correct (rolling average)')
        # write cycle onset
        for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
            ax[2].text(onset+((offset-onset)/2), 50, f'Cycle {i+1}', ha='center', va='center', fontsize=10)

        ax[4].set_xlabel('Trials (groups of 4, with overlapping)')

        cycles_onset.append(data['uid'].max())
        ax[idx].set_xticks(cycles_onset)

    participant = sources[0]["participant"]

    fig.text(0.98, 0.02, participant, ha='right', va='bottom', fontsize=12)

    handles = []
    labels = []
    for axis in ax:
        hand, labe = axis.get_legend_handles_labels()
        handles = hand + handles
        labels = labe + labels

    # for "assessment (generalization words)", use a custom lagend handle
    custom_label = 'assessment (generalization words)'
    if custom_label in labels:
        legend_element = Line2D(
            [0], [0],
            marker='o',
            color='black',
            markerfacecolor='none',
            markersize=4,
            linestyle='None',
            label=custom_label)
        # find index of "assessment (generalization words)"
        idx = labels.index(custom_label)
        handles[idx] = legend_element

    # Remove duplicates from handles and labels
    handles, labels = zip(*set(zip(handles, labels)))

    custom_order = [
        'teaching',
        'assessment (teaching words)',
        'assessment (generalization words)',
        'assessment (MPP)'
    ]

    plt.tight_layout()

    if number_of_rows == 1:
        top_padding = 0.7
    else:
        top_padding = 0.9

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8,
        loc='upper center')
    plt.subplots_adjust(top=top_padding)

    if filter is not None:
        str_filter = '_'.join(filter)
        participant = f'{participant}_{str_filter}'

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'hits_by_trial_{participant}.pdf')
    plt.close()
    cd('..')
    cd('..')

marker_label_map = {
    marker1: 'teaching',
    marker2: 'assessment (teaching words)',
    marker3: 'assessment (generalization words)',
    marker4: 'assessment (MPP)'
}

def hits_by_trial_subplots(data: pd.DataFrame, sources: list):
    def draw_hits(ax: plt.Axes, group: pd.DataFrame, vertical_lines: list):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # draw horizontal line for 0%, 50% of trials, 100%
        for y in [0, 50, 100]:
            ax.axhline(y=y, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['rolling_percentage_no_overlap'],
                marker=marker.replace('_', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        ax.set_ylim([-20, 120])
        ax.set_yticks([0, 50, 100])

    number_of_rows =len(data['relation'].unique())
    unique_sessions = data['session'].unique()

    figsize = (5, 7)
    fig, ax = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=figsize,
        sharex=False, sharey=True)

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        # reset uid
        group['uid'] = range(1, len(group) + 1)
        vertical_lines = []
        cycles_onset = []
        if relation ==  'CD':
            for __, g in group.groupby(['cycle', 'condition', 'session']):
                if g['condition'].unique()[0] == '7':
                    vertical_lines.append(g['uid'].values[0])
                    vertical_lines.append(g['uid'].values[-1])

                    if g['session'].unique()[0] == unique_sessions[-1]:
                        continue
                    else:
                        cycles_onset.append(g['uid'].values[-1])

            cycles_onset.insert(0, vertical_lines[0])
            cycles_onset.append(group['uid'].max())
            ax[idx].set_xticks(cycles_onset)
            # remove first and last vertical line
            vertical_lines.sort()
            vertical_lines = vertical_lines[1:-1]

            for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
                ax[idx].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

        if relation == 'AB':
            for __, g in group.groupby(['cycle', 'condition', 'session']):
                if g['condition'].unique()[0] == '1':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())

            for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                ax[idx].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

            ax[idx].set_xticks(vertical_lines)
            vertical_lines = vertical_lines[1:-1]

        if relation ==  'AC':
            for __, g in group.groupby(['cycle', 'condition', 'session']):
                if g['condition'].unique()[0] == '2a':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())

            for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                ax[idx].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

            ax[idx].set_xticks(vertical_lines)
            vertical_lines = vertical_lines[1:-1]

        if (relation ==  'BC') or (relation == 'CB'):
            for __, g in group.groupby(['cycle', 'condition', 'session']):
                if g['condition'].unique()[0] == '3':
                    vertical_lines.append(g['uid'].values[0])

            vertical_lines.append(group['uid'].max())
            for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                ax[idx].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

            ax[idx].set_xticks(vertical_lines)
            vertical_lines = vertical_lines[1:-1]

        draw_hits(ax[idx], group, vertical_lines)

    ax[-1].set_xlabel('Trials (groups of 4, with overlapping)')
    ax[2].set_ylabel('Percent correct (rolling average)')

    participant = sources[0]["participant"]

    fig.text(0.02, 0.02, participant, ha='left', va='bottom', fontsize=12)
    mapping = {
        'AB' : 'listen to word then select picture',
        'AC' : 'listen to word then select written word',
        'BC' : 'observe picture then select written word',
        'CB' : 'observe written word then select picture',
        'CD' : 'oral reading/textual behavior'
    }
    for idx, (relation, group) in enumerate(data.groupby('relation')):
        # Get the position of the axes in figure coordinates
        ax[idx].text(
            0,
            150,
            f'{relation} - {mapping[relation]}',
            ha='left',
            va='center',
            fontsize=9)

    labels = [
        'teaching',
        'assessment (teaching words)',
        'assessment (generalization words)',
        'assessment (multiple probes procedure)'
    ]

    handles = []
    handles = [
        Line2D([0], [0],
            marker=marker1.replace('_', ''),
            color='gray',
            markerfacecolor='gray',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker1]),
        Line2D([0], [0],
            marker=marker2.replace('_', ''),
            color='black',
            markerfacecolor='black',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker2]),
        Line2D([0], [0],
            marker=marker3.replace('_', ''),
            color='black',
            markerfacecolor='none',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker3]),
        Line2D([0], [0],
            marker=marker4.replace('_', ''),
            color='black',
            markerfacecolor='black',
            markersize=4,
            linestyle='None',
            label=marker_label_map[marker4]),
    ]

    plt.tight_layout()

    top_padding = 0.9
    fig.legend(
        handles,
        labels,
        ncol=2,
        fontsize=8,
        loc='upper center')
    plt.subplots_adjust(top=top_padding)

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'hits_by_trial_{participant}.pdf')
    plt.close()
    cd('..')
    cd('..')


def durations_by_trial(sources: list):
    marker1 = '.'
    marker2 = '._'
    marker3 = '.-'

    def durations_from_sources(sources: list):
        data = []
        for source in sources:
            for relation in source['data']:
                for d in source['data'][relation]:
                    data.append(d)

        result = []
        for source in sources:
            data = []
            for relation in source['data']:
                for d in source['data'][relation]:
                    data.append(d)
            result.append(
                calculate_rolling(
                    pd.DataFrame(data),
                    result_column='duration',
                    percentage=False))

        df = pd.concat(result, axis=0, ignore_index=True)
        color_map = {
            '1': 'gray',
            '2a': 'gray',
            '2b': 'gray',
            '3': 'black',
            '4': 'black',
            '5': 'black',
            '6': 'black',
            '7': 'black'
        }
        df['edgecolors'] = df['condition'].map(color_map)

        facecolor_map = {
            '1': 'gray',
            '2a': 'gray',
            '2b': 'gray',
            '3': 'black',
            '4': 'black',
            '5': 'black',
            '6': 'black',
            '7': 'black'
        }
        df['facecolors'] = df['condition'].map(facecolor_map)

        marker_map = {
            '1': marker1,
            '2a': marker1,
            '2b': marker1,
            '3': marker2,
            '4': marker3,
            '5': marker3,
            '6': marker3,
            '7': marker3
        }
        df['marker'] = df['condition'].map(marker_map)

        # normalize session column
        df['file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        unique_sorted = df['file_num'].sort_values().unique()
        mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        df['session'] = df['file_num'].map(mapping)
        df = df.drop(['file_num', 'file'], axis=1)

        return df

    def draw_durations(ax: plt.Axes, group: pd.DataFrame, vertical_lines: list):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # draw horizontal line for 0%, 50% of trials, 100%
        for y in [0.0]:
            ax.axhline(
                y=y,
                color='black',
                linewidth=0.5,
                linestyle='--', alpha=0.4)

        marker_label_map = {
            marker1: 'teaching',
            marker2: 'assessment (teaching words)',
            marker3: 'assessment (recombinative reading)'
        }

        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['rolling_percentage'],
                marker=marker.replace('_', '').replace('-', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        # ax.set_ylim([-20, 120])
        # ax.set_yticks([0, 50, 100])

        relation = group['relation'].unique()[0]

        mapping = {
            'AB' : 'listen to word then select picture',
            'AC' : 'listen to word then select written word',
            'BC' : 'observe picture then select written word',
            'CB' : 'observe written word then select picture',
            'CD' : 'oral reading/textual behavior'
        }
        # Get the position of the axes in figure coordinates
        ax.text(
            0.0, 1.1,
            f'{relation} - {mapping[relation]}',
            transform=ax.transAxes,
            ha='left', va='center', fontsize=9)

        # calculate trials per minutes in group
        trials = group['uid'].count()
        trials_per_minute = trials/(np.sum(group['duration'])/60)
        relation = group['relation'].unique()[0]
        # Get the position of the axes in figure coordinates
        ax.text(
            1.0, 1.0,
            f'{trials_per_minute:.2f} tpm',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10)

    calculate(sources)
    data = durations_from_sources(sources)
    number_of_rows =len(data['relation'].unique())
    unique_sessions = data['session'].unique()

    cycles_onset = []
    vertical_lines = []
    for __, g in data.groupby(['cycle', 'condition', 'session']):
        if g['condition'].unique()[0] == '7':
            vertical_lines.append(g['uid'].values[0])
            vertical_lines.append(g['uid'].values[-1])
            if g['session'].unique()[0] == unique_sessions[-1]:
                continue
            else:
                cycles_onset.append(g['uid'].values[-1])

        # if g['condition'].unique()[0] == '2b':
        #     vertical_lines.append(g['uid'].values[-1])

    # remove first and last vertical line
    vertical_lines.sort()
    vertical_lines = vertical_lines[1:-1]

    fig, ax = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=(8, 5),
        sharex=True, sharey=True)

    for idx, (_, group) in enumerate(data.groupby('relation')):
        draw_durations(ax[idx], group, vertical_lines)

    ax[2].set_ylabel('Duration (seconds, rolling average)')
    # write cycle onset
    for i, onset in enumerate(cycles_onset):
        ax[0].text(
            onset+80, 10,
            f'Cycle {i+1}',
            ha='left', va='center', fontsize=10)

    ax[4].set_xlabel('Trials (groups of 4,, with overlapping)')

    cycles_onset.append(data['uid'].max())
    ax[idx].set_xticks(cycles_onset)

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    participant = sources[0]["participant"]

    # Collect handles and labels from ax[0] and ax[2]
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[2].get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Remove duplicates from handles and labels
    handles, labels = zip(*set(zip(handles, labels)))
    plt.tight_layout()

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8,
        loc='upper center')
    plt.subplots_adjust(top=0.9)

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'duration_by_trial_{participant}.pdf')
    plt.close()
    cd('..')
    cd('..')

def switchings_by_trial(sources: list):
    marker1 = '.'
    marker2 = '._'
    marker3 = '.-'

    def durations_from_sources(sources: list):
        data = []
        for source in sources:
            for relation in source['data']:
                for d in source['data'][relation]:
                    data.append(d)

        result = []
        for source in sources:
            data = []
            for relation in source['data']:
                for d in source['data'][relation]:
                    data.append(d)
            result.append(
                calculate_rolling(
                    pd.DataFrame(data),
                    result_column='duration',
                    percentage=False))

        df = pd.concat(result, axis=0, ignore_index=True)
        color_map = {
            '1': 'gray',
            '2a': 'gray',
            '2b': 'gray',
            '3': 'black',
            '4': 'black',
            '5': 'black',
            '6': 'black',
            '7': 'black'
        }
        df['edgecolors'] = df['condition'].map(color_map)

        facecolor_map = {
            '1': 'gray',
            '2a': 'gray',
            '2b': 'gray',
            '3': 'black',
            '4': 'black',
            '5': 'black',
            '6': 'black',
            '7': 'black'
        }
        df['facecolors'] = df['condition'].map(facecolor_map)

        marker_map = {
            '1': marker1,
            '2a': marker1,
            '2b': marker1,
            '3': marker2,
            '4': marker3,
            '5': marker3,
            '6': marker3,
            '7': marker3
        }
        df['marker'] = df['condition'].map(marker_map)

        # normalize session column
        df['file_num'] = df['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        unique_sorted = df['file_num'].sort_values().unique()
        mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        df['session'] = df['file_num'].map(mapping)
        df = df.drop(['file_num', 'file'], axis=1)

        return df

    def draw_switchings(ax: plt.Axes, group: pd.DataFrame, vertical_lines: list):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # draw horizontal line for 0%, 50% of trials, 100%
        for y in [0.0]:
            ax.axhline(
                y=y,
                color='black',
                linewidth=0.5,
                linestyle='--', alpha=0.4)

        marker_label_map = {
            marker1: 'teaching',
            marker2: 'assessment (teaching words)',
            marker3: 'assessment (recombinative reading)'
        }

        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['switchings'],
                marker=marker.replace('_', '').replace('-', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        # ax.set_ylim([-20, 120])
        # ax.set_yticks([0, 50, 100])

        relation = group['relation'].unique()[0]

        mapping = {
            'AB' : 'listen to word then select picture',
            'AC' : 'listen to word then select written word',
            'BC' : 'observe picture then select written word',
            'CB' : 'observe written word then select picture',
            'CD' : 'oral reading/textual behavior'
        }
        # Get the position of the axes in figure coordinates
        ax.text(
            0.0, 1.1,
            f'{relation} - {mapping[relation]}',
            transform=ax.transAxes,
            ha='left', va='center', fontsize=9)

        # calculate trials per minutes in group
        trials = group['uid'].count()
        trials_per_minute = trials/(np.sum(group['duration'])/60)
        relation = group['relation'].unique()[0]
        # Get the position of the axes in figure coordinates
        ax.text(
            1.0, 1.0,
            f'{trials_per_minute:.2f} tpm',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10)

    calculate(sources)
    data = durations_from_sources(sources)
    number_of_rows =len(data['relation'].unique())
    unique_sessions = data['session'].unique()

    cycles_onset = []
    vertical_lines = []
    for __, g in data.groupby(['cycle', 'condition', 'session']):
        if g['condition'].unique()[0] == '7':
            vertical_lines.append(g['uid'].values[0])
            vertical_lines.append(g['uid'].values[-1])
            if g['session'].unique()[0] == unique_sessions[-1]:
                continue
            else:
                cycles_onset.append(g['uid'].values[-1])

        # if g['condition'].unique()[0] == '2b':
        #     vertical_lines.append(g['uid'].values[-1])

    # remove first and last vertical line
    vertical_lines.sort()
    vertical_lines = vertical_lines[1:-1]

    fig, ax = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=(8, 5),
        sharex=True, sharey=True)

    for idx, (_, group) in enumerate(data.groupby('relation')):
        draw_switchings(ax[idx], group, vertical_lines)

    ax[2].set_ylabel('Switchings')
    # write cycle onset
    for i, onset in enumerate(cycles_onset):
        ax[0].text(
            onset+80, 10,
            f'Cycle {i+1}',
            ha='left', va='center', fontsize=10)

    ax[4].set_xlabel('Trials (groups of 4,, with overlapping)')

    cycles_onset.append(data['uid'].max())
    ax[idx].set_xticks(cycles_onset)

    fig.text(0.98, 0.98, sources[0]['participant'], ha='right', va='top', fontsize=12)

    participant = sources[0]["participant"]

    # Collect handles and labels from ax[0] and ax[2]
    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles2, labels2 = ax[2].get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Remove duplicates from handles and labels
    handles, labels = zip(*set(zip(handles, labels)))
    plt.tight_layout()

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8,
        loc='upper center')
    plt.subplots_adjust(top=0.9)

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'switchings_by_trial_{participant}.pdf')
    plt.close()
    cd('..')
    cd('..')

def heatmap_by_trial_single(data: pd.DataFrame, sources: list, filter=None):
    def draw_hits(ax: plt.Axes, group: pd.DataFrame, vertical_lines: list):
        # remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # draw vertical dashed line for cycle onset
        for line_x in vertical_lines:
            ax.axvline(x=line_x, color='black', linestyle='--', linewidth=0.5, alpha=0.4)

        # draw horizontal line for 0%, 50% of trials, 100%
        for y in [0, 50, 100]:
            ax.axhline(y=y, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

        marker_label_map = {
            marker1: 'teaching',
            marker2: 'assessment (teaching words)',
            marker3: 'assessment (generalization words)',
            marker4: 'assessment (MPP)'
        }

        for marker, g in group.groupby('marker'):
            ax.scatter(
                g['uid'],
                g['rolling_percentage'],
                marker=marker.replace('_', ''),  # Single marker style for this group
                s=10,
                facecolors=g['facecolors'],
                edgecolors=g['edgecolors'],
                linewidths=0.5,
                label=marker_label_map[marker]  # Optional: Add a label for the legend
            )

        ax.set_ylim([-20, 120])
        ax.set_yticks([0, 50, 100])

        relation = group['relation'].unique()[0]

        mapping = {
            'AB' : 'listen to word then select picture',
            'AC' : 'listen to word then select written word',
            'BC' : 'observe picture then select written word',
            'CB' : 'observe written word then select picture',
            'CD' : 'oral reading/textual behavior'
        }
        # Get the position of the axes in figure coordinates
        ax.text(-15, 150, f'{relation} - {mapping[relation]}', ha='left', va='center', fontsize=9)

    if filter is not None:
        data = data[data['relation'].isin(filter)]

    number_of_rows =len(data['relation'].unique())
    unique_sessions = data['session'].unique()

    if number_of_rows == 1:
        figsize = (5, 2)
    else:
        figsize = (8, 5)
        cycles_onset = []
        vertical_lines = []
        for __, g in data.groupby(['cycle', 'condition', 'session']):
            if g['condition'].unique()[0] == '7':
                vertical_lines.append(g['uid'].values[0])
                vertical_lines.append(g['uid'].values[-1])
                if g['session'].unique()[0] == unique_sessions[-1]:
                    continue
                else:
                    cycles_onset.append(g['uid'].values[-1])

            # if g['condition'].unique()[0] == '2b':
            #     vertical_lines.append(g['uid'].values[-1])

        # remove first and last vertical line
        vertical_lines.sort()
        vertical_lines = vertical_lines[1:-1]

    fig, ax = plt.subplots(
        nrows=number_of_rows,
        ncols=1,
        figsize=figsize,
        sharex=True, sharey=True)

    if number_of_rows == 1:
        ax = [ax]

    for idx, (relation, group) in enumerate(data.groupby('relation')):
        if number_of_rows == 1:
            # reset uid
            group['uid'] = range(1, len(group) + 1)
            vertical_lines = []
            cycles_onset = []
            if relation ==  'CD':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '7':
                        vertical_lines.append(g['uid'].values[0])
                        if filter[0] in ['CD', 'AB']:
                            vertical_lines.append(g['uid'].values[-1])

                        if g['session'].unique()[0] == unique_sessions[-1]:
                            continue
                        else:
                            cycles_onset.append(g['uid'].values[-1])

                cycles_onset.append(group['uid'].max())
                ax[idx].set_xticks(cycles_onset)
                # remove first and last vertical line
                vertical_lines.sort()
                vertical_lines = vertical_lines[1:-1]

                for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

            if relation == 'AB':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '1':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())

                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

            if relation ==  'AC':
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '2a':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())

                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

            if (relation ==  'BC') or (relation == 'CB'):
                for __, g in group.groupby(['cycle', 'condition', 'session']):
                    if g['condition'].unique()[0] == '3':
                        vertical_lines.append(g['uid'].values[0])

                vertical_lines.append(group['uid'].max())
                for i, (onset, offset) in enumerate(itertools.pairwise(vertical_lines)):
                    ax[0].text(onset+((offset-onset)/2), 120, f'C{i+1}', ha='center', va='center', fontsize=10)

                ax[idx].set_xticks(vertical_lines)
                vertical_lines = vertical_lines[1:-1]

        draw_hits(ax[idx], group, vertical_lines)

    if number_of_rows == 1:
        ax[0].set_xlabel('Trials (groups of 4, with overlapping)')
        ax[0].set_ylabel('Percent correct\n(rolling average)')
    else:
        ax[2].set_ylabel('Percent correct (rolling average)')
        # write cycle onset
        for i, (onset, offset) in enumerate(pairwise_non_overlapping(vertical_lines)):
            ax[2].text(onset+((offset-onset)/2), 50, f'Cycle {i+1}', ha='center', va='center', fontsize=10)

        ax[4].set_xlabel('Trials (groups of 4, with overlapping)')

        cycles_onset.append(data['uid'].max())
        ax[idx].set_xticks(cycles_onset)

    participant = sources[0]["participant"]

    fig.text(0.98, 0.02, participant, ha='right', va='bottom', fontsize=12)

    handles = []
    labels = []
    for axis in ax:
        hand, labe = axis.get_legend_handles_labels()
        handles = hand + handles
        labels = labe + labels

    # for "assessment (generalization words)", use a custom lagend handle
    custom_label = 'assessment (generalization words)'
    if custom_label in labels:
        legend_element = Line2D(
            [0], [0],
            marker='o',
            color='black',
            markerfacecolor='none',
            markersize=4,
            linestyle='None',
            label=custom_label)
        # find index of "assessment (generalization words)"
        idx = labels.index(custom_label)
        handles[idx] = legend_element

    # Remove duplicates from handles and labels
    handles, labels = zip(*set(zip(handles, labels)))

    custom_order = [
        'teaching',
        'assessment (teaching words)',
        'assessment (generalization words)',
        'assessment (MPP)'
    ]

    plt.tight_layout()

    if number_of_rows == 1:
        top_padding = 0.7
    else:
        top_padding = 0.9

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=8,
        loc='upper center')
    plt.subplots_adjust(top=top_padding)

    if filter is not None:
        str_filter = '_'.join(filter)
        participant = f'{participant}_{str_filter}'

    # Show the plot
    data_dir()
    cd('analysis')
    cd('output')
    plt.savefig(f'heatmap_b_by_trial_{participant}.pdf')
    plt.close()
    cd('..')
    cd('..')

if __name__ == '__main__':
    from explorer import load_participant_sources

    # for participant in sources:
    #     target_sources = []
    #     for design_file in sources[participant]:
    #         for source in sources[participant][design_file]:
    #             # if 'Sondas-CD-Palavras-12-ensino-8-generalizacao' in source['name']:
    #             if 'Treino-AC-Ref-Intermitente' in source['name']:
    #                 # print(f'  - {source["date"]}, {source["cycle"]}, {source["code"]}')
    #                 target_sources.append(source)

    #     draw_heatmaps_by_trial(target_sources)
    #     # draw_non_uniform_heatmaps(target_sources)

    #     # draw_trial_duration(target_sources)
    #     # draw_switchings(target_sources)
    #     # draw_y_velocity(target_sources)


    sources = load_participant_sources()
    for participant in sources:
        target_sources = []
        for design_file in sources[participant]:
            for source in sources[participant][design_file]:
                if 'Pre-treino' not in source['name']:
                    target_sources.append(source)

        calculate(target_sources)
        data = hits_from_sources(target_sources)
        # durations_by_trial(target_sources)
        hits_by_trial_subplots(data, target_sources)
        # for relation in [['AB'], ['AC'], ['BC'], ['CB'], ['CD']]:
        #     hits_by_trial_single(data, target_sources, relation)
        #     heatmap_by_trial_single(data, target_sources, relation)