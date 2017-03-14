# Written using Anaconda with Python 3.5
# Pate Motter
# 1-19-17

# Input:
#   Timings files should be csv w/ no space
#   Timings cols = num_procs,matrix_name,solver,preconditioner,status,time,iterations,final_residual
#   Timing file examples can be found within the trilinos-prediction/data directory

import pandas as pd
import numpy as np

# Read in the timings from each csv file, aliases for cols
np1_timings = pd.read_csv('../data/janus/janus_np1_results.csv', header=0)
np2_timings = pd.read_csv('../data/janus/janus_np2_results.csv', header=0)
np4_timings = pd.read_csv('../data/janus/janus_np4_results.csv', header=0)
np6_timings = pd.read_csv('../data/janus/janus_np6_results.csv', header=0)
np8_timings = pd.read_csv('../data/janus/janus_np8_results.csv', header=0)
np10_timings = pd.read_csv('../data/janus/janus_np10_results.csv', header=0)
np12_timings = pd.read_csv('../data/janus/janus_np12_results.csv', header=0)

# Make a list of all the individual np dataframes and combine them
timings = [np1_timings, np2_timings, np4_timings, np6_timings, np8_timings, np10_timings, np12_timings]
all_timing_data = pd.concat(timings)
all_timing_data.columns = ['system', 'np', 'matrix', 'solver', 'prec', 'status', 'time', 'iters', 'resid']

# Change string entries into numerical (for SKLearn)
all_timing_data['system_id'] = all_timing_data.system.map(
    {'janus': 0, 'bridges': 1, 'comet': 2}).astype(int)
all_timing_data['solver_id'] = all_timing_data.solver.map(
    {'FIXED_POINT': 0, 'BICGSTAB': 1, 'MINRES': 2, 'PSEUDOBLOCK_CG': 3, 'PSEUDOBLOCK_STOCHASTIC_CG': 4,
     'PSEUDOBLOCK_TFQMR': 5, 'TFQMR': 6, 'LSQR': 7, 'PSEUDOBLOCK_GMRES': 8}).astype(int)
all_timing_data['prec_id'] = all_timing_data.prec.map({'ILUT': 0, 'RILUK': 1, 'RELAXATION': 2, 'CHEBYSHEV': 3,
                                                       'NONE': 4}).astype(int)
all_timing_data['status_id'] = all_timing_data.status.map({'error': -1, 'unconverged': 0, 'converged': 1}).astype(int)

# Group based on the matrix, find the best times for each matrix (error, unconverged, converged)
grouped = all_timing_data.groupby(['matrix', 'status_id'])
matrix_best_times = grouped['time'].aggregate(np.min)

# Create two empty lists that will be new columns
good_bad_list = []
new_time_list = []
hash_list = []

hash_dict = {}
matrix_names = all_timing_data['matrix'].unique()
for name in matrix_names:
    hash_dict[name] = hash(name)

# Iterate through each row of the dataframe
subset = all_timing_data[['time', 'matrix', 'status_id']]
max_float_value = np.finfo(np.float32).max

for index, row in subset.iterrows():
    current_matrix_time = row['time']
    matrix_name = row['matrix']
    hash_list.append(hash_dict[matrix_name])

    # Check for matrices which never converged
    try:
        matrix_min_time = matrix_best_times[matrix_name][1]  # 1 indicates converged
    except:
        matrix_min_time = np.inf

    # Error or unconverged runs = max float time
    if row['status_id'] != 1:
        good_bad_list.append(-1)
        new_time_list.append(np.inf)
    # Good = anything within 25% of the fastest run for that matrix
    elif current_matrix_time <= 1.25 * matrix_min_time:
        good_bad_list.append(1)
        new_time_list.append(current_matrix_time)
    # Bad = anything else outside of that range but still converged
    else:
        good_bad_list.append(-1)
        new_time_list.append(current_matrix_time)

# Create Pandas series from the lists which used to contain strings
good_bad_series = pd.Series(good_bad_list)
new_time_series = pd.Series(new_time_list)
name_hash_series = pd.Series(hash_list)

# Add the series to the dataframe as columns
all_timing_data = all_timing_data.assign(good_or_bad=pd.Series(good_bad_series))
all_timing_data = all_timing_data.assign(new_time=pd.Series(new_time_series))
all_timing_data = all_timing_data.assign(matrix_id=pd.Series(name_hash_series))

# Select which columns to keep and output to file
cleaned_timing_data = all_timing_data[['system_id', 'np', 'matrix', 'matrix_id', 'solver_id', 'prec_id',
                                       'status_id', 'new_time', 'good_or_bad']]
cleaned_timing_data.to_csv('janus_processed_timings.csv')
all_timing_data.to_csv('janus_unprocessed_timings.csv')

