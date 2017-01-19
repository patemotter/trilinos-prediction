import pandas as pd
import numpy as np

# Read in the timings from each csv file, aliases for cols
np1_timings = pd.read_csv('../data/np1_results.csv', header=0)
np1_timings.columns = ['np', 'matrix', 'solver', 'prec', 'status', 'time', 'iters', 'resid']

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
combined = pd.merge(np1_timings, properties, on='matrix')

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

# Iterate through each row of the dataframe
for index, row in combined.iterrows():
    current_matrix_time = row['time']
    matrix_name = row['matrix']

    # Check for matrices which never converged
    try:
        matrix_min_time = matrix_best_times[matrix_name][1]  # 1 indicates converged
    except:
        matrix_min_time = np.inf

    # Error or unconverged runs = inf time
    if row['status_num'] != 1:
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

# Create Pandas series from the lists
good_bad_series = pd.Series(good_bad_list)
good_bad_series.to_csv('good_bad_series.csv')
new_time_series = pd.Series(new_time_list)
new_time_series.to_csv('new_time_series.csv')

# Add series to dataframe as new columns
np1_timings = np1_timings.assign(good_or_bad=pd.Series(good_bad_series))
np1_timings = np1_timings.assign(new_time=pd.Series(new_time_series))
np1_timings = np1_timings.assign(solver_num=combined['solver_num'])
np1_timings = np1_timings.assign(prec_num=combined['prec_num'])
np1_timings = np1_timings.assign(status_num=combined['status_num'])

np1_timings = np1_timings.drop(['matrix', 'solver', 'prec', 'status', 'iters', 'resid'], axis=1)

np1_timings.to_csv('revised_np1_results.csv')

# combined = combined.assign(good_or_bad=pd.Series(good_bad_series))
# combined = combined.assign(new_time=pd.Series(new_time_series))
# combined.to_csv('combined_np1.csv')
