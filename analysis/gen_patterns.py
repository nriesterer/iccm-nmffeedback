""" Generates the patterns (W and H matrices) from JNMF application.

"""

import sys
import os
import time

import numpy as np
import pandas as pd
import ccobra
import matplotlib.pyplot as plt
import seaborn as sns

import batchprocjnmf as nmf


def df_to_matrix(df):
    """ Converts a CCOBRA dataset into matrix form.

    """

    clean = df[["id", "task", "response"]]
    usr = list(clean["id"].unique())
    matrix = np.zeros((576, len(usr)))

    for _, row in clean.iterrows():
        usr_idx = usr.index(row["id"])
        syl_item = ccobra.Item(usr_idx, "syllogistic", row["task"], "single-choice", "", 0)
        syllog = ccobra.syllogistic.Syllogism(syl_item)
        enc_resp = syllog.encode_response(row["response"].split(";"))

        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syllog.encoded_task)
        resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
        comb_idx = syl_idx * 9 + resp_idx

        if matrix[comb_idx, usr_idx] != 0:
            print("Tried to write twice to field")
            exit()
        matrix[comb_idx, usr_idx] = 1

    return matrix

def get_response_vector(pattern):
    """ Obtain the response vector from a pattern.

    """

    result = []
    prediction_matrix = pattern.reshape(64, 9)
    for i in range(64):
        result.append(ccobra.syllogistic.RESPONSES[prediction_matrix[i].argmax()])
    return result

def evaluate(data_df, model_pattern, factor=1.0):
    """ Evaluate a model pattern on the data.

    """

    model_pred = get_response_vector(model_pattern)

    result = []
    for subj, subj_df in data_df.groupby("id"):
        for _, task in subj_df.iterrows():
            task_list = [x.split(";") for x in task["task"].split("/")]
            resp_list = task["response"].split(";")
            task_enc = ccobra.syllogistic.encode_task(task_list)
            resp_enc = ccobra.syllogistic.encode_response(resp_list, task_list)

            pred = model_pred[ccobra.syllogistic.SYLLOGISMS.index(task_enc)]
            hit = factor if (resp_enc == pred) else 0

            result.append(hit)
    return result

def get_model_performance(data1_df, data2_df, models):
    """ Obtain model performances.

    """

    result = []
    # High group model
    result.extend(evaluate(data1_df, models[0]))
    result.extend(evaluate(data1_df, models[1], factor=0.25))
    result.extend(evaluate(data2_df, models[1], factor=0.25))
    result.extend(evaluate(data2_df, models[2]))
    return np.mean(result)

def criterion(X1, X2, W_high, H_high, W_low, H_low, data1_df, data2_df, perf_weight=2):
    """ Grid search performance optimization criterion.

    """

    data1_pattern = W_high[:,1]
    data2_pattern = W_low[:,1]
    common_pattern = (W_high[:,0] + W_low[:,0]) / 2
    models = [data1_pattern, common_pattern, data2_pattern]

    # Total Model performance
    model_perf = 1 - get_model_performance(data1_df, data2_df, models)
    difference_error = np.sum(data1_pattern * data2_pattern)
    common_error = 1 - np.sum(W_high[:,0] * W_low[:,0])

    error1 = nmf.reconstruction_error(X1, W_high, H_high) / (X1.shape[0] * X1.shape[1])
    error2 = nmf.reconstruction_error(X2, W_low, H_low) / (X2.shape[0] * X2.shape[1])

    return perf_weight * model_perf + 0.5 * (difference_error + common_error) + (error1 + error2)

def plot_patterns(data1_pattern, common_pattern, data2_pattern, d1_name, d2_name):
    """ Plot the patterns.

    """

    # reshape patterns to matrices
    data1_pattern = data1_pattern.reshape(64, 9)
    common_pattern = common_pattern.reshape(64, 9)
    data2_pattern = data2_pattern.reshape(64, 9)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(8, 10), sharey=True)
    #fig.suptitle("Different patterns", fontsize=18)
    sns.set(style='darkgrid')

    # axes for all 3 subplots
    high_ax = axs[0]
    common_ax = axs[1]
    low_ax = axs[2]

    every = 2

    # plot high group
    high_ax.set_title('{}'.format(d1_name.replace("ccobra_", "").replace('exp3_', '').replace('_full', '')))
    sns.heatmap(data1_pattern,
                ax=high_ax,
                cmap=sns.color_palette('Blues', 100),
                cbar=False,
                linewidths=0.5)
    high_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90)
    high_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    high_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0)

    for _, spine in high_ax.spines.items():
        spine.set_visible(True)

    # plot common group
    common_ax.set_title('Common pattern')
    sns.heatmap(common_pattern,
                ax=common_ax,
                cmap=sns.color_palette('Purples', 100),
                cbar=False,
                linewidths=0.5)
    common_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90)
    common_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    common_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0)
    common_ax.tick_params(axis='y', which='both', length=0)

    for _, spine in common_ax.spines.items():
        spine.set_visible(True)

    # plot low group
    low_ax.set_title('{}'.format(d2_name.replace("ccobra_", "").replace('exp3_', '').replace('_full', '')))
    sns.heatmap(data2_pattern,
                ax=low_ax,
                cmap=sns.color_palette('Reds', 100),
                cbar=False,
                linewidths=0.5)
    low_ax.set_xticklabels(ccobra.syllogistic.RESPONSES, rotation=90)
    low_ax.set_yticks(np.arange(0, 64, every) + 0.5)
    low_ax.set_yticklabels(ccobra.syllogistic.SYLLOGISMS[::every], rotation=0)
    low_ax.tick_params(axis='y', which='both', length=0)

    for _, spine in low_ax.spines.items():
        spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/{}_{}_contrast.pdf".format(d1_name, d2_name))
    plt.show()


if __name__== "__main__":
    if len(sys.argv) != 3:
        print("Usage python ccobra_nmf.py [dataset1] [dataset2]")
        exit()

    data_df1 = pd.read_csv(sys.argv[1])
    data_df2 = pd.read_csv(sys.argv[2])

    d1_name = os.path.basename(sys.argv[1]).replace(".csv", "")
    d2_name = os.path.basename(sys.argv[2]).replace(".csv", "")

    print("Names: {}, {}".format(d1_name, d2_name))

    # create matrices
    X1 = df_to_matrix(data_df1)
    X2 = df_to_matrix(data_df2)

    print("Input matrices:")
    print("    X1", X1.shape)
    print("    X2", X2.shape)
    print()

    alphas = [75, 250, 350, 500, 750, 1000]
    betas = [75, 250, 350, 500, 750, 1000]

    best_score = 10000
    best = None
    num_samples = 3

    for sample in range(num_samples):
        print("Starting run {}/{}".format(sample + 1, num_samples))
        for alpha in alphas:
            print("    alpha: {}".format(alpha))
            for beta in betas:
                start = time.time()
                print("        beta: {}".format(beta))
                W1, H1, W2, H2 = nmf.factorize(X1, X2, 1, 1,
                                               alpha=alpha, beta=beta, lr=0.01,
                                               outer_loops=20,
                                               inner_loops=500,
                                               silent=True)
                crit = criterion(X1, X2, W1, H1, W2, H2, data_df1, data_df2)
                if crit <= best_score:
                    best_score = crit
                    best = (alpha, beta, W1, H1, W2, H2)
                    t = "            Found better fit ({}) with (alpha={}, beta={})"
                    print(t.format(np.round(crit, 2), alpha, beta))

                    np.save("fit_results/fit_{}_{}_result_W_{}.npy".format(d1_name, d2_name, d1_name), W1)
                    np.save("fit_results/fit_{}_{}_result_W_{}.npy".format(d1_name, d2_name, d2_name), W2)
                    np.save("fit_results/fit_{}_{}_result_H_{}.npy".format(d1_name, d2_name, d1_name), H1)
                    np.save("fit_results/fit_{}_{}_result_H_{}.npy".format(d1_name, d2_name, d2_name), H2)
                print("            finished after {:2f}s".format(time.time() - start))
    print("Finished, overall best: alpha={}, beta={}".format(best[0], best[1]))

    # best values
    alpha, beta, d1_W, d1_H, d2_W, d2_H = best

    # total H sum
    d1_sum = np.sum(d1_H, axis=0)
    d2_sum = np.sum(d2_H, axis=0)

    # extract common vector
    c1 = d1_W[:,0]
    c2 = d2_W[:,0]
    common_importance = (d1_sum[0] + d2_sum[0]) / np.sum(d1_sum + d2_sum)
    common_pattern = (c1 + c2) / 2

    # discriminative vectors
    d1_pattern = d1_W[:,1]
    d2_pattern = d2_W[:,1]

    distinct_importance = (d1_sum[1] + d2_sum[1]) / np.sum(d1_sum + d2_sum)
    d1_importance = d1_sum[1] / np.sum(d1_sum)
    d2_importance = d2_sum[1] / np.sum(d2_sum)

    print("Stats")
    print("-----")
    print("Common error:", 1 - np.sum(c1 * c2))
    print("Difference error:", np.sum(d1_pattern * d2_pattern))
    print()
    print("Common importance:", common_importance)
    print("Difference importance:", distinct_importance)
    print("    {} pattern importance:".format(d1_name), d1_importance)
    print("    {} pattern importance:".format(d2_name), d2_importance)

    plot_patterns(d1_pattern, common_pattern, d2_pattern, d1_name, d2_name)
