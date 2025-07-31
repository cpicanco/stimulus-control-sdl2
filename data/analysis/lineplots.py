import numpy as np
import matplotlib.pyplot as plt

from barplots import default_axis_config, save_or_show

def line_plot_per_cycle(container, save=False, include_names=[], append_to_filename='', group=False):
    """
    container = [{
        categories : [(name, hit_rate, color), ...],
        identification : [(name, date, duration, cycle), ...]}
        names : ['BC Probes 2', 'CB Probes 2', 'AC Probes', 'CD Probes 2']
    """
    # group all hit_rates by name
    grouped_hit_rates = {}
    max_cycle = 0
    for data in container:
        identification = data['identification']
        categories = data['categories']
        # we need to group hit_rates per name then per cycle
        # cycle is the last element in identification tuple
        id_names, id_data = zip(*identification)
        max_cycle = max(max_cycle, int(id_data[3]))
        cycle = id_data[3]
        if cycle == '0':
            continue
        cycle = 'Cycle ' + cycle
        names, hit_rates, colors = zip(*categories)

        for name, hit_rate in zip(names, hit_rates):
            if name not in include_names:
                continue

            if hit_rate is None:
                continue

            if cycle in grouped_hit_rates:
                if name in grouped_hit_rates[cycle]:
                    if isinstance(hit_rate, (int, float)):
                        hit_rate = [hit_rate]
                    grouped_hit_rates[cycle][name].extend(hit_rate)
                else:
                    grouped_hit_rates[cycle][name] = [hit_rate]
            else:
                grouped_hit_rates[cycle] = {name: [hit_rate]}

    # group 'BC Probes 2' and 'CB Probes 2' hit rates
    if 'BC Probes 2' in include_names and 'CB Probes 2' in include_names:
        for cycle, hit_rates in grouped_hit_rates.items():
            if 'BC Probes 2' in hit_rates and 'CB Probes 2' in hit_rates:
                hit_rates['BC-CB Probes'] = hit_rates['BC Probes 2'] + hit_rates['CB Probes 2']
                # remove 'BC Probes 2' and 'CB Probes 2' from the dictionary
                del hit_rates['BC Probes 2']
                del hit_rates['CB Probes 2']

    # remove 'BC Probes 2' and 'CB Probes 2' from include_names
    include_names = [name for name in include_names if name not in ['BC Probes 2', 'CB Probes 2']]
    include_names.append('BC-CB Probes')

    # now we should have names == ['BC-CB Probes', 'AC Probes', 'CD Probes 2']
    # grouped_hit_rates = {'Cycle 1': {'BC-CB Probes': [0.5, 0.6], 'AC Probes': [0.7], 'CD Probes 2': [0.8, 0.9]}, ...}

    # calculate average hit rate and sdt per cycle per name
    average_hit_rates = {}
    std_hit_rates = {}
    for cycle, hit_rates in grouped_hit_rates.items():
        for name, rates in hit_rates.items():
            if name not in average_hit_rates:
                average_hit_rates[name] = []
                std_hit_rates[name] = []
            average_hit_rates[name].append(np.mean(rates))
            std_hit_rates[name].append(np.std(rates))

    # we need a single plot with cycles at the x axis and hit_rates at the y axis, one line per name
    fig, ax = plt.subplots(figsize=(6, 3))
    default_axis_config(ax, False)
    ax.set_xlim(0, max_cycle + 1)
    ax.set_ylim(-10, 120)
    ax.set_xticks(range(1, max_cycle + 1))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Percent correct (Group main)')

    # plot one line per name for all cycles
    handles = []
    for name, hit_rates in average_hit_rates.items():
        # add a little bit of space in x direction to avoid overlapping lines
        x = np.array(range(1, max_cycle + 1)) + 0.08 * include_names.index(name)

        # use a circle for CD Probes 2, a triangle for AC Probes and a square for BC-CB Probes
        if name == 'CD Probes 2':
            label = 'Textual Beh'
            marker = 'o'
            markerfacecolor = 'gray'
        elif name == 'AC Probes':
            label = 'SW-PW MTS'
            marker = '^'
            markerfacecolor = 'black'
        elif name == 'BC-CB Probes':
            label = 'Pic-PW MTS'
            marker = 's'
            markerfacecolor = 'white'
        else:
            raise ValueError('Unknown name: {}'.format(name))

        # draw a dashed line at 0.5 and 1.0
        ax.axhline(y=50, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(y=100, color='black', linewidth=0.8, linestyle='--')

        hit_rates = [rate * 100 for rate in hit_rates]
        if group:
            upper_errors = std_hit_rates[name]
            lower_errors = [0] * len(upper_errors)
            upper_errors = [rate * 100 for rate in upper_errors]

            # Draw the error bars
            for i in range(len(x)):
                ax.vlines(x[i], hit_rates[i] - lower_errors[i], hit_rates[i] + upper_errors[i],
                        linewidth=1.0,
                        color='black')

            # Draw the caps
            for i in range(len(x)):
                ax.hlines(hit_rates[i] + upper_errors[i], x[i] - 0.1, x[i] + 0.1,
                        linewidth=1.0,
                        color='black')

        handle, = ax.plot(x, hit_rates,
                    markersize=8,
                    label=label,
                    color='black',
                    marker=marker,
                    linewidth=1.0,
                    markerfacecolor=markerfacecolor, markeredgecolor='black')
        handles.append(handle)

    # y ticks at 0, 0.5 and 1.0 only
    yticks = [0, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1f}'.format(y) for y in yticks])
    # show legend outside the plot
    handles = [handles[1], handles[2], handles[0]]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    # tight layout
    plt.tight_layout()

    filename = 'Fig_line_per_cycle'+append_to_filename+'.pdf'
    save_or_show(fig, save, filename)

def line_plot_per_cycle_df(df, include_names=[], append_to_filename='', group=False):

    # calculate average hit rate and sdt per cycle per name
    cd_probes = df[df['Condition'] == 5 ]

    # we need a single plot with cycles at the x axis and hit_rates at the y axis, one line per name
    fig, ax = plt.subplots(figsize=(6, 3))
    default_axis_config(ax, False)
    ax.set_xlim(0, max_cycle + 1)
    ax.set_ylim(-10, 120)
    ax.set_xticks(range(1, max_cycle + 1))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Percent correct (Group main)')

    # plot one line per name for all cycles
    handles = []
    for name, hit_rates in average_hit_rates.items():
        # add a little bit of space in x direction to avoid overlapping lines
        x = np.array(range(1, max_cycle + 1)) + 0.08 * include_names.index(name)

        # use a circle for CD Probes 2, a triangle for AC Probes and a square for BC-CB Probes
        if name == 'CD Probes 2':
            label = 'Textual Beh'
            marker = 'o'
            markerfacecolor = 'gray'
        elif name == 'AC Probes':
            label = 'SW-PW MTS'
            marker = '^'
            markerfacecolor = 'black'
        elif name == 'BC-CB Probes':
            label = 'Pic-PW MTS'
            marker = 's'
            markerfacecolor = 'white'
        else:
            raise ValueError('Unknown name: {}'.format(name))

        # draw a dashed line at 0.5 and 1.0
        ax.axhline(y=50, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(y=100, color='black', linewidth=0.8, linestyle='--')

        hit_rates = [rate * 100 for rate in hit_rates]
        if group:
            upper_errors = std_hit_rates[name]
            lower_errors = [0] * len(upper_errors)
            upper_errors = [rate * 100 for rate in upper_errors]

            # Draw the error bars
            for i in range(len(x)):
                ax.vlines(x[i], hit_rates[i] - lower_errors[i], hit_rates[i] + upper_errors[i],
                        linewidth=1.0,
                        color='black')

            # Draw the caps
            for i in range(len(x)):
                ax.hlines(hit_rates[i] + upper_errors[i], x[i] - 0.1, x[i] + 0.1,
                        linewidth=1.0,
                        color='black')

        handle, = ax.plot(x, hit_rates,
                    markersize=8,
                    label=label,
                    color='black',
                    marker=marker,
                    linewidth=1.0,
                    markerfacecolor=markerfacecolor, markeredgecolor='black')
        handles.append(handle)

    # y ticks at 0, 0.5 and 1.0 only
    yticks = [0, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.1f}'.format(y) for y in yticks])
    # show legend outside the plot
    handles = [handles[1], handles[2], handles[0]]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    # tight layout
    plt.tight_layout()

    filename = 'lines_per_cycle'+append_to_filename+'.pdf'
    save_or_show(fig, False, filename)

if __name__ == '__main__':
    from explorer import load_all_from_csv

    df = load_all_from_csv()
    include = ['CD', 'AC', 'BC', 'CB']
    line_plot_per_cycle_df(
        df,
        include_names=include,
        append_to_filename='_'.join(include),
        group=True)