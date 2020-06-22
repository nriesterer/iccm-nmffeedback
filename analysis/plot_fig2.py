""" Plots figure 2.

"""

from matplotlib.lines import Line2D
import ccobra
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_response_vector(pattern):
    """ Extracts a vector of responses from a given pattern by identifying the response with
    maximum weight for each syllogistic problem.

    """

    result = []
    prediction_matrix = pattern.reshape(64, 9)
    for i in range(64):
        result.append(ccobra.syllogistic.RESPONSES[prediction_matrix[i].argmax()])
    return result

def evaluate_all_models(datasets, models):
    """ Evaluates all models for the given datasets. Produces a data table containing
    information about the data-table combination and evaluation results (i.e., predictions, hits,
    etc.).

    """

    result = []

    # Iterate over models
    for model in models:
        model_name = ("{} model ({} - {})" \
            .format(model["model_target"], model["model_datasets"][0], model["model_datasets"][1])) \
            .replace("ccobra_", "")
        model_pattern = model["model_pattern"]
        model_pred = get_response_vector(model_pattern)
        model_group = model["model_target"]
        model_origin = model["model_datasets"]

        # Iterate over datasets
        for dataset_name, dataset_content in datasets.items():
            data_df = dataset_content["data"]

            # Iterate over subjects contained in the dataset
            for subj, subj_df in data_df.groupby("id"):

                # Query model for individual predictions and compare with truth to obtain hits
                for _, task in subj_df.iterrows():
                    task_list = [x.split(";") for x in task["task"].split("/")]
                    resp_list = task["response"].split(";")
                    task_enc = ccobra.syllogistic.encode_task(task_list)
                    resp_enc = ccobra.syllogistic.encode_response(resp_list, task_list)

                    pred = model_pred[ccobra.syllogistic.SYLLOGISMS.index(task_enc)]
                    hit = (resp_enc == pred)

                    # Store the evaluation result
                    result.append({
                        "dataset" : dataset_name,
                        "model" : model_name.replace('exp3_', '').replace('_full', ''),
                        "model_group" : model_group,
                        "model_origin" : (x.replace('exp3_', '').replace('_full', '') for x in model_origin),
                        "subj" : subj,
                        "task" : task_enc,
                        "truth" : resp_enc,
                        "pred" : pred,
                        "hit" : hit
                    })

    return pd.DataFrame(result)


# Load the datasets
datasets = {
    "control": {
        "fname": "ccobra_control",
        "data": pd.read_csv("data/ccobra_control.csv")
    },
    "1 sec": {
        "fname": "ccobra_1s",
        "data": pd.read_csv("data/ccobra_1s.csv")
    },
    "10 sec": {
        "fname": "ccobra_10s",
        "data": pd.read_csv("data/ccobra_10s.csv")
    }
}

names = [x["fname"] for x in datasets.values()]

# Prepare colors for plotting
dashes = ["dashed", "dotted", "solid"]
common_colors = sns.color_palette("Greys_d", 3)
other_colors = [
    sns.color_palette("Blues_d", 2),
    sns.color_palette("Reds_d", 2),
    sns.color_palette("Greens_d", 2)
]

# Prepare plot settings
hue_order = []
palette = {}
models = []

# Iterate over filename indices
for i in range(len(names) - 1):

    # Iterate over remaining filename indices
    for j in range(i + 1, len(names)):
        # Load the JNMF data corresponding to the filenames
        template = "fit_results/fit_{}_{}_result_W_{}.npy" # dataset1, dataset2, matrix, dataset
        W_lower_feedback = np.load(template.format(names[i], names[j], names[i]))
        W_higher_feedback = np.load(template.format(names[i], names[j], names[j]))

        # Extract the common pattern as the average between both common patterns
        common = (W_lower_feedback[:,0] + W_higher_feedback[:,0]) / 2

        # Store the patterns as models
        models.append({
            "model_datasets" : (names[i], names[j]),
            "model_target" : "common",
            "model_pattern" : common
        })

        models.append({
            "model_datasets" : (names[i], names[j]),
            "model_target" : names[i],
            "model_pattern" : W_lower_feedback[:,1]
        })

        models.append({
            "model_datasets" : (names[i], names[j]),
            "model_target" : names[j],
            "model_pattern" : W_higher_feedback[:,1]
        })

        # Update plot color options
        palette[("common model ({} - {})".format(names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', '')] = common_colors[(i + j + 2) % 3]
        palette[("{} model ({} - {})".format(names[i], names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', '')] = other_colors[i][j - 1]
        palette[("{} model ({} - {})".format(names[j], names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', '')] = other_colors[j][i]

        hue_order.append(("common model ({} - {})".format(names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', ''))
        hue_order.append(("{} model ({} - {})".format(names[i], names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', ''))
        hue_order.append(("{} model ({} - {})".format(names[j], names[i], names[j])).replace("ccobra_", "").replace('exp3_', '').replace('_full', ''))

# Evaluate the models
result_df = evaluate_all_models(datasets, models)
print(result_df.head())

# Prepare the plot
sns.set(style='whitegrid', palette="colorblind")
f = plt.figure(figsize=(10, 3.5))
ax = plt.gca()

hue_order = [
    'common model (control - 1s)',
    'common model (control - 10s)',
    'common model (1s - 10s)',
    'control model (control - 1s)',
    'control model (control - 10s)',
    '1s model (control - 1s)',
    '1s model (1s - 10s)',
    '10s model (control - 10s)',
    '10s model (1s - 10s)',
]

# Plot the data
cp = sns.pointplot(x="dataset", y="hit", hue="model", data=result_df, ax=ax, palette=palette, hue_order=hue_order)
cp.set(xlabel='Dataset', ylabel='Accuracy')

# Plot a custom legend
legend_els = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['common model (control - 1s)'], markersize=10, label='(control - 1s) $\\rightarrow$ common'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['common model (control - 10s)'], markersize=10, label='(control - 10s) $\\rightarrow$ common'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['common model (1s - 10s)'], markersize=10, label='(1s - 10s) $\\rightarrow$ common'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['control model (control - 1s)'], markersize=10, label='(control - 1s) $\\rightarrow$ control'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['control model (control - 10s)'], markersize=10, label='(control - 10s) $\\rightarrow$ control'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['1s model (control - 1s)'], markersize=10, label='(control - 1s) $\\rightarrow$ 1s'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['1s model (1s - 10s)'], markersize=10, label='(1s - 10s) $\\rightarrow$ 1s'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['10s model (control - 10s)'], markersize=10, label='(control - 10s) $\\rightarrow$ 10s'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['10s model (1s - 10s)'], markersize=10, label='(1s - 10s) $\\rightarrow$ 10s'),
]

ax.legend(title="Models",
    handles=legend_els, frameon=True, loc='center left',
    ncol=1, bbox_to_anchor=(1, 0.5), mode='None'
)

# Store and show the image
plt.tight_layout()
plt.savefig('results/all_models.pdf')
plt.show()
