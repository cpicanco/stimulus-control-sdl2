import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def default_axis_config(ax, limit_y=True):
    if limit_y:
        ax.set_ylim(-.02, 1.1)
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_yticklabels(np.arange(0, 1.2, 0.2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

def bar_plot(categories, identification):
    """
        categories : [(name, hit_rate, color), ...]
        identification : [(name, data), ...]
    """
    names, hit_rates, colors = zip(*categories)
    id_names, id_data = zip(*identification)

    fig, ax = plt.subplots(figsize=(10, 5))
    default_axis_config(ax)
    ax.bar(names, hit_rates, color=colors)

    # add identification into the plot
    ax.text(0.5, 1.1, f'{id_names[0]} {id_data[0]}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.5, 1.05, f'{id_names[1]} {id_data[1]}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.5, 1.0, f'{id_names[2]} {id_data[2]}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    formatter = ticker.FormatStrFormatter('%.1f')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel('Category')
    plt.ylabel('Hit Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def bar_subplots(container):
    """
    container = [{categories : [(name, hit_rate, color), ...], identification : [(name, data), ...]} ]
    """
    # decide how many columns and rows we need using the contaner length
    number_of_plots = len(container)
    columns = 2
    rows = number_of_plots // columns + 1

    fig, axs = plt.subplots(rows, columns, figsize=(5, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    n = number_of_plots
    bottom_plots = [n - 2, n - 1] if n % 2 == 0 else [n - 1]


    for ax in axs[number_of_plots:]:
        fig.delaxes(ax)

    for i, (data, ax) in enumerate(zip(container, axs[:number_of_plots])):
        categories = data['categories']
        identification = data['identification']
        names, hit_rates, colors = zip(*categories)
        id_names, id_data = zip(*identification)

        # ax = axs[i // columns, i % columns]
        default_axis_config(ax)
        ax.bar(names, hit_rates, color=colors)
        ax.set_title(f'{id_names[0]} {id_data[0]}\n{id_names[1]} {id_data[1]}\n{id_names[2]} {id_data[2]}', fontsize=8)

        formatter = ticker.FormatStrFormatter('%.1f')
        ax.yaxis.set_major_formatter(formatter)

        ax.set_xticklabels(names, rotation=45, ha='right')
        if i in bottom_plots:
            ax.tick_params(axis='x', labelbottom=True)


    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()