import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from fileutils import data_dir, cd
from player_media import draw_word, load_fonts
from player_information import GazeInfo
from player_synchronizer import Synchronizer

K = 12.031777217921098/4

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

def get_section_cols(of_type : str, section: str):
    cols_count = []
    cols_duration = []
    if section == 'samples':
        section = 'sample'
    elif section == 'comparisons':
        section == 'comparison'
    else:
        raise ValueError('section must be samples or comparisons')

    if of_type == 'positive' or section == 'samples':
        iterator = range(1, 2)
    elif of_type == 'negative':
        iterator = range(2, 4)
    elif of_type == 'simulteneous_sample':
        iterator = range(4, 5)
    else:
        raise ValueError('of_type must be positive, negative, or simultaneous_sample')

    for i in iterator:
        for j in range(1, 5):
            cols_count.append(
                f'{section}{i}_letter{j}_fixations_count'
            )

            cols_duration.append(
                f'{section}{i}_letter{j}_fixations_duration'
            )
    return cols_count, cols_duration

if __name__ == '__main__':
    from dataclasses import dataclass
    from enum import Enum

    from player_utils import calculate
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    import pandas as pd

    from explorer_study2 import (
        load_data,
        export
    )

    cols_duration_by_count = [
        'letter1_duration_by_count',
        'letter2_duration_by_count',
        'letter3_duration_by_count',
        'letter4_duration_by_count',
    ]

    cols_duration_times_count = [
        'letter1_duration_times_count',
        'letter2_duration_times_count',
        'letter3_duration_times_count',
        'letter4_duration_times_count',
    ]

    # create enum type for comparisons and samples
    class Section(Enum):
        Samples = 'sample'
        Comparisons = 'comparisons'

    class ComparisonsType(Enum):
        Positive = 'positive'
        Negative = 'negative'
        SimultaneousSample = 'simulteneous_sample'

    class Relation(Enum):
        AB = 'AB'
        AC = 'AC'
        CD = 'CD'
        BC = 'BC'
        BB = 'BB'
        CB = 'CB'

    class CalculationType(Enum):
        NONE = "none"
        ROLLING_SUM = "rolling_sum_no_overlap"
        RATIO = "ratio"
        PRODUCT = "product"

    @dataclass
    class Metric:
        name : str
        header : str
        relation : Relation
        source_columns: list[str]  # Columns needed for calculation
        target_columns: list[str]  # Result columns
        calculation: CalculationType
        percentage: bool = False   # Parameter for calculations that need it

    def get_section_cols(of_type : str, section: Section):
        cols_count = []
        cols_duration = []
        if section not in Section:
            raise ValueError(f'section must be {", ".join([s.value for s in Section])}')

        if of_type == ComparisonsType.Positive or section == Section.Samples:
            iterator = range(1, 2)
        elif of_type == ComparisonsType.Negative:
            iterator = range(2, 4)
        elif of_type == ComparisonsType.SimultaneousSample:
            iterator = range(4, 5)
        else:
            raise ValueError(f'of_type must be {", ".join([s.value for s in ComparisonsType])}')

        for i in iterator:
            for j in range(1, 5):
                basename = f'{section.value}{i}_letter{j}'
                cols_count.append(
                    f'{basename}_fixations_count'
                )

                cols_duration.append(
                    f'{basename}_fixations_duration'
                )
        return cols_count, cols_duration

    def process_metrics(data: pd.DataFrame, metrics: list[Metric]) -> pd.DataFrame:
        """Process all metrics in a single loop with structured operations"""
        processed_data = []
        for metric in metrics:
            # Apply filtering here
            df = data[data['relation'] == metric.relation.value].copy()
            # Handle rolling sum calculations
            processed_metric = []
            for src_col, tgt_col in zip(metric.source_columns, metric.target_columns):
                if metric.calculation == CalculationType.NONE:
                    processed_metric.append(df[src_col].fillna(0))
                if metric.calculation == CalculationType.ROLLING_SUM:
                    processed_metric.append(calculate(
                        df,
                        calculation=metric.calculation.value,
                        target_column=src_col,
                        percentage=metric.percentage
                    )['target_calculation'])

                elif metric.calculation == CalculationType.RATIO:
                    df[tgt_col] = (
                        df[src_col[0]] / df[src_col[1]]
                    )
                    processed_metric.append(df[tgt_col].fillna(0))

                elif metric.calculation == CalculationType.PRODUCT:
                    # Handle product calculations: source[0] * source[1]
                    df[tgt_col] = (
                        df[src_col[0]] * df[src_col[1]]
                    )
                    processed_metric.append(df[tgt_col].fillna(0))
                # Add new calculation types here...

            df = pd.concat(processed_metric, axis=1)
            processed_data.append(df)
        return processed_data

    METRICS = []
    # CD
    cols_count, cols_duration = get_section_cols(
        of_type=ComparisonsType.Positive,
        section=Section.Samples)
    pairwise_src_columns = [cols for cols in zip(cols_duration, cols_count)]
    METRICS.append(
        Metric(
            name='duration_by_count',
            header='Duration\nby Count',
            relation=Relation.CD,
            source_columns=pairwise_src_columns,
            target_columns=cols_duration_by_count,
            calculation=CalculationType.RATIO
        )
    )

    # # AC
    # cols_count, cols_duration = get_section_cols(
    #     of_type=ComparisonsType.Positive,
    #     section=Section.Comparisons)
    # pairwise_src_columns = [cols for cols in zip(cols_duration, cols_count)]
    # METRICS.append(
    #     Metric(
    #         name='duration_by_count',
    #         header='Duration\nby Count',
    #         relation=Relation.AC,
    #         source_columns=pairwise_src_columns,
    #         target_columns=cols_duration_by_count,
    #         calculation=CalculationType.RATIO
    #     )
    # )

    # # BC
    # cols_count, cols_duration = get_section_cols(
    #     of_type=ComparisonsType.Positive,
    #     section=Section.Comparisons)
    # pairwise_src_columns = [cols for cols in zip(cols_duration, cols_count)]
    # METRICS.append(
    #     Metric(
    #         name='duration_by_count',
    #         header='Duration\nby Count',
    #         relation=Relation.BC,
    #         source_columns=pairwise_src_columns,
    #         target_columns=cols_duration_by_count,
    #         calculation=CalculationType.RATIO
    #     )
    # )

    # # CB

    # cols_count, cols_duration = get_section_cols(
    #     of_type=ComparisonsType.Positive,
    #     section=Section.Samples)
    # pairwise_src_columns = [cols for cols in zip(cols_duration, cols_count)]
    # METRICS.append(
    #     Metric(
    #         name='sample_duration_by_count',
    #         header='Duration\nby Count (Sample, CB)',
    #         relation=Relation.CB,
    #         source_columns=pairwise_src_columns,
    #         target_columns=cols_duration_by_count,
    #         calculation=CalculationType.RATIO
    #     )
    # )

    # cols_count, cols_duration = get_section_cols(
    #     of_type=ComparisonsType.SimultaneousSample,
    #     section=Section.Comparisons)
    # pairwise_src_columns = [cols for cols in zip(cols_duration, cols_count)]
    # METRICS.append(
    #     Metric(
    #         name='comparisons_duration_by_count',
    #         header='Duration\nby Count (Comparisons, CB)',
    #         relation=Relation.CB,
    #         source_columns=pairwise_src_columns,
    #         target_columns=cols_duration_by_count,
    #         calculation=CalculationType.RATIO
    #     ),
    # )

    df = load_data()
    df['participant'] = df.apply(lambda row: row['participant'].replace('\\', '').replace('-', '_'), axis=1)
    # normalize session column
    i = 0
    for participant, participant_data in df.groupby('participant'):
        participant_data['file_num'] = participant_data['file'].str.extract(r'^(\d+)\.data\.processed$').astype(int)
        unique_sorted = participant_data['file_num'].sort_values().unique()
        session_mapping = {num: new_num for new_num, num in enumerate(unique_sorted)}
        participant_data['session'] = participant_data['file_num'].map(session_mapping) + 1

        processed_data = process_metrics(participant_data, METRICS)

        fig, axs = plt.subplots(nrows=1, ncols=len(processed_data), figsize=(6, 10), sharey=True)

        fig.text(0.5, 0.95, participant)

        # test if axs is a list
        if isinstance(axs, plt.Axes):
            axs = [axs]
        else:
            axs = axs.flatten()

        for ax, data, metric in zip(axs, processed_data, METRICS):
            heatmap_data = data.to_numpy()
            heatmaps_max = heatmap_data.max()
            im = ax.imshow(
                heatmap_data,
                aspect='auto',  # Ensure rows and columns scale proportionally
                extent=[0, 4*K, len(heatmap_data), 0],
                vmin=0,
                vmax=heatmaps_max,
                cmap='viridis'
            )

            # Add labels
            ax.set_xlabel('Letters')
            if metric.name == 'duration':
                ax.set_ylabel('Trials')
            ax.set_xticks([i*K for i in range(5)])



















            
            # ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'])

            # Add colorbar horizontally on top
            cbar = plt.colorbar(im, ax=ax, location='top', orientation='horizontal', shrink=0.8, aspect=50)
            cbar.ax.xaxis.set_ticks_position('top')  # Move ticks to the top
            cbar.ax.xaxis.set_label_position('top')  # Move label to the top
            cbar.set_label(metric.name)
            # Format the colorbar to show 3 decimal places
            if 'duration' in metric.name:
                cbar.formatter = FormatStrFormatter('%.3f')  # Set format to 3 decimal places

            cbar.update_ticks()  # Update ticks to apply formatting

        plt.tight_layout()
        plt.show()
