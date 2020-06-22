""" Plots figure 1.

"""

import sys

import ccobra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Prepare the color dictionary
color_dict = {
    "common" : sns.color_palette("Greys", 100),
    "Control" : sns.color_palette("Blues", 100),
    "1s" : sns.color_palette("Reds", 100),
    "10s" : sns.color_palette("Greens", 100)
}

def plot_patterns(data1_pattern, common_pattern, data2_pattern, d1_name, d2_name, common_importance, d1_importance, d2_importance):
    """ Plot the patterns.

    """

    fname = "results/{}_{}_contrast.pdf".format(d1_name, d2_name)
    d1_name = d1_name.replace("ccobra_", "").replace('exp3_', '').replace('_full', '')
    d2_name = d2_name.replace("ccobra_", "").replace('exp3_', '').replace('_full', '')
    d1_name = d1_name[0].upper() + d1_name[1:]
    d2_name = d2_name[0].upper() + d2_name[1:]
    template = "{} pattern\n({})"

    # reshape patterns to matrices
    data1_pattern = data1_pattern.reshape(64, 9)
    common_pattern = common_pattern.reshape(64, 9)
    data2_pattern = data2_pattern.reshape(64, 9)

    # Prepare for plotting
    fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)
    sns.set(style='darkgrid')

    # Axes for all 3 subplots
    high_ax = axs[0]
    common_ax = axs[1]
    low_ax = axs[2]

    every = 2

    # Plot high group
    high_ax.set_title(template.format(d1_name, np.round(d1_importance, 2)), fontsize=16)
    sns.heatmap(data1_pattern,
                ax=high_ax,
                cmap=color_dict[d1_name],
                cbar=False,
                linewidths=0.5)
    high_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90, fontsize=14)
    high_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    high_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0, fontsize=14)

    for _, spine in high_ax.spines.items():
        spine.set_visible(True)

    # Plot common group
    common_ax.set_title(template.format("Common", np.round(common_importance, 2)), fontsize=16)
    sns.heatmap(common_pattern,
                ax=common_ax,
                cmap=color_dict["common"],
                cbar=False,
                linewidths=0.5)
    common_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90, fontsize=14)
    common_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    common_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0)
    common_ax.tick_params(axis='y', which='both', length=0)

    for _, spine in common_ax.spines.items():
        spine.set_visible(True)

    # Plot low group
    low_ax.set_title(template.format(d2_name, np.round(d2_importance, 2)), fontsize=16)
    sns.heatmap(data2_pattern,
                ax=low_ax,
                cmap=color_dict[d2_name],
                cbar=False,
                linewidths=0.5)
    low_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90, fontsize=14)
    low_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    low_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0)
    low_ax.tick_params(axis='y', which='both', length=0)

    for _, spine in low_ax.spines.items():
        spine.set_visible(True)

    # Store and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname)
    plt.show()

if __name__== "__main__":
    if len(sys.argv) != 3:
        print("Usage python plot_patterns.py [dataset1] [dataset2]")
        exit()

    # Extract command line arguments
    d1_name = sys.argv[1]
    d2_name = sys.argv[2]

    # Load JNMF patterns
    template = "fit_results/fit_{}_{}_result_{}_{}.npy"
    d1_W = np.load(template.format(d1_name, d2_name, "W", d1_name))
    d1_H = np.load(template.format(d1_name, d2_name, "H", d1_name))
    d2_W = np.load(template.format(d1_name, d2_name, "W", d2_name))
    d2_H = np.load(template.format(d1_name, d2_name, "H", d2_name))

    print("Names: {}, {}".format(d1_name, d2_name))

    # Total H sum
    d1_sum = np.sum(d1_H, axis=0)
    d2_sum = np.sum(d2_H, axis=0)

    # Extract common vector
    c1 = d1_W[:,0]
    c2 = d2_W[:,0]
    common_importance = ((d1_sum[0] /  np.sum(d1_sum)) + (d2_sum[0] /  np.sum(d2_sum))) / 2
    common_pattern = (c1 + c2) / 2

    # Discriminative vectors
    d1_pattern = d1_W[:,1]
    d2_pattern = d2_W[:,1]

    d1_importance = d1_sum[1] / np.sum(d1_sum)
    d2_importance = d2_sum[1] / np.sum(d2_sum)

    print("Stats")
    print("-----")
    print("Common error:", 1 - np.sum(c1 * c2))
    print("Difference error:", np.sum(d1_pattern * d2_pattern))
    print()
    print("Common importance:", common_importance)
    print("Difference importance:", (d1_importance + d2_importance) / 2)
    print("    {} pattern importance:".format(d1_name), d1_importance)
    print("    {} pattern importance:".format(d2_name), d2_importance)

    # Generate the plot
    plot_patterns(d1_pattern, common_pattern, d2_pattern, d1_name, d2_name, common_importance, d1_importance, d2_importance)
