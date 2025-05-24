import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from player_information import GazeInfo
from player_synchronizer import Synchronizer

from fileutils import change_data_folder_name, data_dir, cd
from study2_constants import foldername

width = 1440
height = 900

grid = {
    0:{'x':916,'y':35,'w':504,'h':230},
    1:{'x':20,'y':35,'w':513,'h':230},
    2:{'x':906,'y':635,'w':514,'h':230},
    3:{'x':20,'y':635,'w':514,'h':230}}

def show_heatmap(source):
    participante = source['participant']
    date = source['date']
    cycle = source['cycle']
    code = source['code']

    change_data_folder_name(foldername)
    data_dir()
    cd(os.path.join(participante, 'analysis', date, cycle))

    session = Synchronizer(code)

    word_filter = session.word_filter() # TODO

    fixations_at_words = word_filter.fixations_by_time()

    heatmap_data, xedges, yedges = np.histogram2d(fixations_at_words['FPOGX'], fixations_at_words['FPOGY'],
                                                bins=[11, 15],
                                                range=[[0, session.screen.width], [0, session.screen.height]],
                                                weights=fixations_at_words['FPOGD'])  # Using fixation duration as weights

    # calculate total duration
    total_duration = fixations_at_words['FPOGD'].sum()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))

    # Add rectangles to the heatmap
    ax = plt.gca()
    for rect in grid.values():
        # Create a rectangle: Rectangle((x, y), width, height)
        rectangle = patches.Rectangle((rect['x'], rect['y']), rect['w'], rect['h'],
                                    linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rectangle)

    # show total duration
    plt.text(0.5, 0.5, f'Total Duration: {total_duration} ms', horizontalalignment='center', verticalalignment='center',
            fontsize=16, color='black', transform=ax.transAxes)
    plt.imshow(heatmap_data.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin='upper', cmap='viridis')
    plt.title('Heatmap of Valid Fixations (Frequency and Duration)')
    plt.xlabel('FPOGX')
    plt.ylabel('FPOGY')
    plt.colorbar(label='Total Fixation Duration')
    plt.show()

if __name__ == '__main__':
    sources = []
    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-26',
        'code': '004',
        'cycle': '1'})

    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-27',
        'code': '012',
        'cycle': '2'})

    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-29',
        'code': '022',
        'cycle': '3'})

    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-29',
        'code': '031',
        'cycle': '4'})

    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-30',
        'code': '040',
        'cycle': '5'})

    sources.append({
        'participant': '26-MSS',
        'date': '2024-08-30',
        'code': '048',
        'cycle': '6'})

    for source in sources:
        show_heatmap(source)