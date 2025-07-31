import numpy as np
from scipy.stats import pearsonr, linregress, kendalltau
import statsmodels.api as sm
import matplotlib.pyplot as plt

from figure_idioms import translations, language

# Assuming you have two lists of data:
def plot_correlation(x, y, xlabel, ylabel, title, save=False, limit_x=False):
    if len(x) == 0 or len(y) == 0:
        print(f"Skipping plot for {title} due to empty data.")
        return

    tau, tau_p_value = kendalltau(x, y)
    print(f"Kendall's tau({len(x)-2}) = {tau:.3f}, p = {tau_p_value:.3f}")

    r, p = pearsonr(x, y)
    print(f"r({len(x)-2}) = {r:.3f}, p = {round(p):.3f}")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(f"Regression Slope: {slope:.3f}")
    print(f"Regression Intercept: {intercept:.3f}")
    print(f"R^2: {r_value**2:.3f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Standard Error: {std_err:.3f}")

    # plot
    fig, ax = plt.subplots(figsize=(4, 3.8))
    # set size
    # ax.set_aspect(1)
    ax.scatter(x, y)
    #draw line of best fit
    ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    ax.text(0.05, 0.9, f'r = {r:.3f}, p < 0.001', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'τ = {tau:.3f}, p = {tau_p_value:.3f}', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    if ylabel == 'Latência':
        plt.ylim(0, 40)

    if limit_x:
        plt.xlim((-.2, .35))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)

    fig.tight_layout()
    if save:
        plt.savefig(f'correlation_pt_br_{title}.svg')
    else:
        plt.show()
    plt.close(fig)

def plot_correlation_2(x1, y1, x2, y2, xlabel, ylabel, name, title1, title2, save=False):
    if len(x1) == 0 or len(y1) == 0 or len(x2) == 0 or len(y2) == 0:
        print(f"Skipping plot for {name} due to empty data.")
        return

    # prepare data for both sets
    correlation_matrix1 = np.corrcoef(x1, y1)
    correlation_xy1 = correlation_matrix1[0,1]
    r_squared1 = correlation_xy1**2
    polyfit1 = np.polyfit(x1, y1, 1)
    slope1 = polyfit1[0]


    correlation_matrix2 = np.corrcoef(x2, y2)
    correlation_xy2 = correlation_matrix2[0,1]
    r_squared2 = correlation_xy2**2
    polyfit2 = np.polyfit(x2, y2, 1)
    slope2 = polyfit2[0]

    # plot
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))  # Create a figure and two subplots
    axs[0].scatter(x1, y1)  # Plot on the first subplot
    axs[1].scatter(x2, y2)  # Plot on the second subplot

    # draw line of best fit on both subplots
    axs[0].plot(np.unique(x1), np.poly1d(polyfit1)(np.unique(x1)), color='red')
    axs[1].plot(np.unique(x2), np.poly1d(polyfit2)(np.unique(x2)), color='red')

    # add text to both subplots
    axs[0].text(0.3, 0.9, f'R^2 = {round(r_squared1, 2)}, Slope = {round(slope1, 2)}', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.3, 0.9, f'R^2 = {round(r_squared2, 2)}, Slope = {round(slope2, 2)}', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)

    if ylabel == translations[language]['correlation.plot_correlation_2']['ylabel']:
        axs[0].set_ylim(0, 40)
        axs[1].set_ylim(0, 40)

    # axs[0].set_xlabel(xlabel)
    axs[0].set_title(title1)

    axs[1].set_xlabel(xlabel)
    axs[1].set_title(title2)

    fig.text(0.0, 0.5, ylabel, va='center', rotation='vertical')
    fig.tight_layout()

    if save:
        plt.savefig(f'{name}_latency.png')
    else:
        plt.show()
    plt.close(fig)