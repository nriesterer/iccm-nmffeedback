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

def similarity(dude1, dude2):
    """ Computes similarity between two reasoners.

    """

    return np.mean(dude1 == dude2)

def intra_similarity(dudes):
    """ Computes intra similarity in a set of reasoners.

    """

    sims = []
    for i in range(len(dudes) - 1):
        for j in range(i + 1, len(dudes)):
            sims.append(similarity(dudes[i], dudes[j]))
    return np.mean(sims)

def closest_intra_similarity(dudes):
    """ Compute closest intra similarity.

    """

    sims = []
    for i in range(len(dudes)):
        best_sim = 0
        for j in range(len(dudes)):
            if i == j:
                continue
            best_sim = max(best_sim, similarity(dudes[i], dudes[j]))
        sims.append(best_sim)
    return np.mean(sims)

def inter_group_similarity(dudes1, dudes2):
    """ Compute inter group similarity.

    """

    sims = []
    for dude1 in dudes1:
        for dude2 in dudes2:
            sims.append(similarity(dude1, dude2))
    return np.mean(sims)

def closest_inter_group_similarity(dudes1, dudes2):
    """ Compute closest inter group similarity.

    """

    sims = []
    for dude1 in dudes1:
        best_sim = 0
        for dude2 in dudes2:
            best_sim = max(best_sim, similarity(dude1, dude2))
        sims.append(best_sim)
    return np.mean(sims)

def assign_to_closest_dataset(dudes, dataset1, dataset2):
    """ Compute assignments to closest dataset.

    """

    dataset1_dudes = []
    dataset2_dudes = []
    equilibrium = []
    for dude in dudes:
        sim_to_d1 = closest_inter_group_similarity([dude], dataset1)
        sim_to_d2 = closest_inter_group_similarity([dude], dataset2)
        if sim_to_d1 > sim_to_d2:
            dataset1_dudes.append(dude)
        elif sim_to_d1 < sim_to_d2:
            dataset2_dudes.append(dude)
        else:
            equilibrium.append(dude)
    return dataset1_dudes, dataset2_dudes, equilibrium


# Load the datasets
control_df = pd.read_csv("data/ccobra_control.csv")
exp_1s_df = pd.read_csv("data/ccobra_1s.csv")
exp_10s_df = pd.read_csv("data/ccobra_10s.csv")

# Convert into matrix form
control_mat = df_to_matrix(control_df)
exp_1s_mat = df_to_matrix(exp_1s_df)
exp_10s_mat = df_to_matrix(exp_10s_df)

# Extract lists of indivdiuals
control_dudes = mat_to_dudes_list(control_mat)
exp_1s_dudes = mat_to_dudes_list(exp_1s_mat)
exp_10s_dudes = mat_to_dudes_list(exp_10s_mat)

# Compute reassignments
control_assignments = [len(x) / len(control_dudes) for x in assign_to_closest_dataset(control_dudes, exp_1s_dudes, exp_10s_dudes)]
exp_1s_assignments = [len(x)  / len(exp_1s_dudes)  for x in assign_to_closest_dataset(exp_1s_dudes, control_dudes, exp_10s_dudes)]
exp_10s_assignments = [len(x) / len(exp_10s_dudes) for x in assign_to_closest_dataset(exp_10s_dudes, control_dudes, exp_1s_dudes)]

# Print reassignment numbers
print("Assignment to closest other group")
print("    control:\n        exp 1s  = {}\n        exp 10s = {}\n        none    = {}"
    .format(control_assignments[0], control_assignments[1], control_assignments[2]))
print()
print("    exp 1s: \n        control = {}\n        exp 10s = {}\n        none    = {}"
    .format(exp_1s_assignments[0], exp_1s_assignments[1], exp_1s_assignments[2]))
print()
print("    exp 10s:\n        control = {}\n        exp 1s  = {}\n        none    = {}"
    .format(exp_10s_assignments[0], exp_10s_assignments[1], exp_10s_assignments[2]))
print()
