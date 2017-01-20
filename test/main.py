# Written using Anaconda with Python 3.5
# Pate Motter
# 1-19-17

# Timings files should be csv w/ no space
# Timings cols = num_procs,matrix_name,solver,preconditioner,status,time,iterations,final_residual
# Timing file examples can be found within the trilinos-prediction/data directory

# Properties files have many columns
# Properties file used is trilinos-prediction/data/uflorida-features.csv

import pandas as pd
import numpy as np

# Read in the timings from each csv file, aliases for cols
np1_timings = pd.read_csv('../data/np1_results.csv', header=0)
np2_timings = pd.read_csv('../data/np2_results.csv', header=0)
np4_timings = pd.read_csv('../data/np4_results.csv', header=0)
np6_timings = pd.read_csv('../data/np6_results.csv', header=0)
np8_timings = pd.read_csv('../data/np8_results.csv', header=0)
np10_timings = pd.read_csv('../data/np10_results.csv', header=0)
np12_timings = pd.read_csv('../data/np12_results.csv', header=0)

timings = [np1_timings, np2_timings, np4_timings, np6_timings, np8_timings, np10_timings, np12_timings]
all_timing_data = pd.concat(timings)
matrix_names = all_timing_data['matrix_name'].unique()
matrix_list = {}
for name in matrix_names:
    matrix_list[hash(name)] = name
    a = pd.Series(matrix_list)

a.to_csv('hashed.csv')

all_timing_data.columns = ['np', 'matrix', 'solver', 'prec', 'status', 'time', 'iters', 'resid']

# Read in the matrix properties from the csv file, aliases cols
properties = pd.read_csv('../data/uflorida-features.csv', header=0)
properties.columns = ['rows', 'cols', 'min_nnz_row', 'row_var', 'col_var',
                      'diag_var', 'nnz', 'frob_norm', 'symm_frob_norm',
                      'antisymm_frob_norm', 'one_norm', 'inf_norm', 'symm_inf_norm',
                      'antisymm_inf_norm', 'max_nnz_row', 'trace', 'abs_trace',
                      'min_nnz_row', 'avg_nnz_row', 'dummy_rows', 'dummy_rows_kind',
                      'num_value_symm_1', 'nnz_pattern_symm_1', 'num_value_symm_2',
                      'nnz_pattern_symm_2', 'row_diag_dom', 'col_diag_dom', 'diag_avg',
                      'diag_sign', 'diag_nnz', 'lower_bw', 'upper_bw', 'row_log_val_spread',
                      'col_log_val_spread', 'symm', 'matrix']

# Combine the two dataframes (timings and properties)
combined = pd.merge(all_timing_data, properties, on='matrix')
print(len(combined))

# Change string entries into numerical (for SKLearn)
combined['solver_num'] = combined.solver.map(
    {'FIXED_POINT': 0, 'BICGSTAB': 1, 'MINRES': 2, 'PSEUDOBLOCK_CG': 3, 'PSEUDOBLOCK_STOCHASTIC_CG': 4,
     'PSEUDOBLOCK_TFQMR': 5, 'TFQMR': 6, 'LSQR': 7, 'PSEUDOBLOCK_GMRES': 8}).astype(int)
combined['prec_num'] = combined.prec.map({'ILUT': 0, 'RILUK': 1, 'RELAXATION': 2, 'CHEBYSHEV': 3,
                                          'NONE': 4}).astype(int)
combined['status_num'] = combined.status.map({'error': -1, 'unconverged': 0, 'converged': 1}).astype(int)

# Group based on the matrix, find the best times for each matrix (error, unconverged, converged)
grouped = combined.groupby(['matrix', 'status_num'])
matrix_best_times = grouped['time'].aggregate(np.min)

# Create two empty lists that will be new columns
good_bad_list = []
new_time_list = []
hash_list = []

total_count = 0
count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
# Iterate through each row of the dataframe
for index, row in combined.iterrows():
    total_count += 1
    current_matrix_time = row['time']
    matrix_name = row['matrix']
    hash_list.append(hash(matrix_name))

    # Check for matrices which never converged
    try:
        matrix_min_time = matrix_best_times[matrix_name][1]  # 1 indicates converged
    except:
        matrix_min_time = np.inf
        count_0 += 1

    # Error or unconverged runs = inf time
    if row['status_num'] != 1:
        good_bad_list.append(-1)
        new_time_list.append(np.inf)
        count_1 += 1
    # Good = anything within 25% of the fastest run for that matrix
    elif current_matrix_time <= 1.25 * matrix_min_time:
        good_bad_list.append(1)
        new_time_list.append(current_matrix_time)
        count_2 += 2
    # Bad = anything else outside of that range but still converged
    else:
        good_bad_list.append(-1)
        new_time_list.append(current_matrix_time)
        count_3 += 3

print(count_0, count_1, count_2, count_3, total_count)

# Create Pandas series from the lists
good_bad_series = pd.Series(good_bad_list)
good_bad_series.to_csv('good_bad_series.csv')
combined = combined.assign(good_or_bad=pd.Series(good_bad_series))

new_time_series = pd.Series(new_time_list)
new_time_series.to_csv('new_time_series.csv')
combined = combined.assign(new_time=pd.Series(new_time_series))

name_hash_series = pd.Series(hash_list)
name_hash_series.to_csv('hash_series.csv')
combined = combined.assign(matrix_hash=pd.Series(name_hash_series))

# Add series to dataframe as new columns
# todo This is causing the main issue, need new way of outputting to files
ready = combined[['np', 'matrix_hash', 'solver_num', 'prec_num', 'status_num', 'new_time', 'good_or_bad']]
ready.to_csv('ready.csv')
