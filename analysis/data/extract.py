import datetime

import pandas as pd
import numpy as np


def decode_quant(enc_quant):
    return enc_quant.replace('A', 'All').replace('I', 'Some').replace('E', 'No').replace('O', 'Some not')

def decode_task(enc_task, A, B, C):
    fig = int(enc_task[-1])
    quant1 = decode_quant(enc_task[0])
    quant2 = decode_quant(enc_task[1])

    if fig == 1:
        return '{};{};{}/{};{};{}'.format(quant1, A, B, quant2, B, C)
    if fig == 2:
        return '{};{};{}/{};{};{}'.format(quant1, B, A, quant2, C, B)
    if fig == 3:
        return '{};{};{}/{};{};{}'.format(quant1, A, B, quant2, C, B)
    if fig == 4:
        return '{};{};{}/{};{};{}'.format(quant1, B, A, quant2, B, C)

    raise ValueError('Invalid figure encountered')

def decode_response(enc_resp, A, C):
    if enc_resp == 'nvc':
        return 'NVC'

    quant = decode_quant(enc_resp[0].upper())
    terms = [quant, A, C] if enc_resp[1:] == 'ac' else [quant, C, A]
    return ';'.join(terms)

def gen_choices(A, C):
    choices = []
    for quant in ['A', 'I', 'E', 'O']:
        choices.append(decode_response(quant + 'ac', A, C))
        choices.append(decode_response(quant + 'ca', A, C))
    choices.append('NVC')
    return '|'.join(choices)

def df_to_ccobra_data(df):
    ccobra_data = []
    for subj, subj_df in df.groupby('token'):
        subj_group = subj_df['subject_group'].unique()
        assert len(subj_group) == 1

        # Ignore subject
        if subj == 'I51yn74':
            print('   Ignoring {} (too slow too often)'.format(subj))
            continue

        # Attention check
        att_check = subj_df.loc[subj_df['part_id'] == 'instructions3_attention_check', 'answer_values'].values
        assert len(att_check) < 2
        if len(att_check) == 0 or att_check[0] != 'reasoning':
            print('   Ignoring {} (attention check)'.format(subj))
            continue

        # Serious participation check
        ser_check = subj_df.loc[subj_df['part_id'] == 'serious_participation', 'answer_values'].values
        assert len(ser_check) < 2
        if len(ser_check) == 0 or ser_check[0] != 'yes':
            print('   Ignoring {} (ser)'.format(subj))
            continue

        # Time check
        start_time = datetime.datetime.strptime(subj_df['time'].values[0][:-4], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.datetime.strptime(subj_df['time'].values[-1][:-4], '%Y-%m-%d %H:%M:%S')
        diff = end_time - start_time
        if diff.total_seconds() / 60 < 20:
            print('   Ignoring {} (<20min)'.format(subj))
            continue

        subj_df = subj_df.loc[subj_df['section'] == 'syllog_tasks']
        seq_idx = 0

        subj_ccobra_data = []
        for taskname, taskname_df in subj_df.groupby('taskname', sort=False):
            if 'sample' in taskname:
                continue

            # Only consider responses given within the time limit
            if np.all(taskname_df['page_timeout_exceeded'] == True):
                continue

            # Extract premise and answer rows
            prem_idx = np.argmax(['premises' in x for x in taskname_df['element_id']])
            assert prem_idx in [0, 1]

            premises_series = taskname_df.iloc[prem_idx]
            answer_series = taskname_df.iloc[1 - prem_idx]
            assert 'premises' in premises_series['element_id']
            assert 'answer' in answer_series['element_id']

            # Extract premise and answer information
            enc_task = premises_series['hidden_syllog']
            placeholder_dict = dict(eval(premises_series['placeholder_values']))
            A, B, C = [placeholder_dict[x] for x in ['A', 'B', 'C']]
            enc_resp = answer_series['answer_values']

            # Skip NAN responses
            if pd.isna(enc_resp):
                # print('missing')
                continue

            # Construct CCOBRA data
            ccobra_task = decode_task(enc_task, A, B, C)
            ccobra_resp = decode_response(enc_resp, A, C)
            ccobra_choices = gen_choices(A, C)

            # Store CCOBRA data
            subj_ccobra_data.append({
                'id': subj,
                'sequence': seq_idx,
                'task': ccobra_task,
                'choices': ccobra_choices,
                'response': ccobra_resp,
                'domain': 'syllogistic',
                'response_type': 'single-choice',
                'group': subj_group[0]
            })

            seq_idx += 1

        ccobra_data.extend(subj_ccobra_data)

    df = pd.DataFrame(ccobra_data)
    return df


# Load data
exp1_df = pd.read_csv('raw/data_exp1.csv', delimiter=',', encoding='utf-8') # 1s vs. no feedback
exp2_df = pd.read_csv('raw/data_exp2.csv', delimiter=',', encoding='utf-8')
exp3_df = pd.read_csv('raw/data_exp3.csv', delimiter=',', encoding='utf-8')

# Convert to CCOBRA data
print('Parsing experiment 1 data...')
ccobra_exp1_df = df_to_ccobra_data(exp1_df)
print('Parsing experiment 2 data...')
ccobra_exp2_df = df_to_ccobra_data(exp2_df)
print('Parsing experiment 3 data...')
ccobra_exp3_df = df_to_ccobra_data(exp3_df)

# Extract the condition subsets
exp1_1s_df = ccobra_exp1_df.loc[ccobra_exp1_df['group'] == 'a']
exp1_control_df = ccobra_exp1_df.loc[ccobra_exp1_df['group'] == 'b']
exp2_1s_df = ccobra_exp2_df
exp3_1s_df = ccobra_exp3_df.loc[ccobra_exp3_df['group'] == 'a']
exp3_10s_df = ccobra_exp3_df.loc[ccobra_exp3_df['group'] == 'b']

ccobra_control_df = exp1_control_df
ccobra_1s_df = pd.concat((exp1_1s_df, exp2_1s_df, exp3_1s_df))
ccobra_10s_df = exp3_10s_df

print('control: {} subjects.'.format(len(ccobra_control_df['id'].unique())))
print('     1s: {} subjects.'.format(len(ccobra_1s_df['id'].unique())))
print('    10s: {} subjects.'.format(len(ccobra_10s_df['id'].unique())))

# Store the datasets
ccobra_control_df.to_csv('ccobra_control.csv', index=False)
ccobra_1s_df.to_csv('ccobra_1s.csv', index=False)
ccobra_10s_df.to_csv('ccobra_10s.csv', index=False)
