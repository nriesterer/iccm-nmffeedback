""" Generates data for table 2.

"""

import ccobra
import numpy as np
import pandas as pd

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

def mat_to_dudes_list(mat):
    """ Extract list of vectors representing response patterns of individual participants.

    """

    return [mat[:,x].reshape(64,9).argmax(axis=1) for x in range(mat.shape[1])]

def calculate_dude_performance_on_tasks(tasks, dude):
    """ Compute the performance of an individual participant.

    """

    hits = 0
    for task in tasks:
        solutions = ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[task]
        task_idx = ccobra.syllogistic.SYLLOGISMS.index(task)
        dude_answer = ccobra.syllogistic.RESPONSES[dude[task_idx]]
        if dude_answer in solutions:
            hits += 1
    return hits / len(tasks)

def calculate_dataset_performance_on_tasks(tasks, dataset):
    """ Compute the performance for dataset of individuals.

    """

    perfs = [calculate_dude_performance_on_tasks(tasks, x) for x in dataset]
    return np.mean(perfs), np.std(perfs, ddof=1) / np.sqrt(len(perfs))


# Load the datasets
control_df = pd.read_csv("data/ccobra_control.csv")
exp_1s_df = pd.read_csv("data/ccobra_1s.csv")
exp_10s_df = pd.read_csv("data/ccobra_10s.csv")

# Transform datasets to matrices
control_mat = df_to_matrix(control_df)
exp_1s_mat = df_to_matrix(exp_1s_df)
exp_10s_mat = df_to_matrix(exp_10s_df)

# Extract list of individuals
control_dudes = mat_to_dudes_list(control_mat)
exp_1s_dudes = mat_to_dudes_list(exp_1s_mat)
exp_10s_dudes = mat_to_dudes_list(exp_10s_mat)

# Print results
print("Total performance")
print("    control: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.SYLLOGISMS, control_dudes)))
print("    exp 1s:  {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.SYLLOGISMS, exp_1s_dudes)))
print("    exp 10s: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.SYLLOGISMS, exp_10s_dudes)))
print()

print("Valid performance")
print("    control: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.VALID_SYLLOGISMS, control_dudes)))
print("    exp 1s:  {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.VALID_SYLLOGISMS, exp_1s_dudes)))
print("    exp 10s: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.VALID_SYLLOGISMS, exp_10s_dudes)))
print()

print("Invalid performance")
print("    control: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.INVALID_SYLLOGISMS, control_dudes)))
print("    exp 1s:  {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.INVALID_SYLLOGISMS, exp_1s_dudes)))
print("    exp 10s: {} +- {}".format(*calculate_dataset_performance_on_tasks(ccobra.syllogistic.INVALID_SYLLOGISMS, exp_10s_dudes)))
print()
