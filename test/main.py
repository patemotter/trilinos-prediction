import pandas as pd
import numpy as np

timings = pd.read_csv("../data/all_results_janus_single_node_1-14-17.csv")
timings.columns = ['np', 'matrix', 'solver', 'prec', 'status', 'time', 'iters', 'resid']
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

matrix_list = timings['matrix'].unique()
combined = pd.merge(timings, properties, on='matrix')
combined['solver_num'] = combined.solver.map(
    {'FIXED_POINT': 0, 'BICGSTAB': 1, 'MINRES': 2, 'PSEUDOBLOCK_CG': 3, 'PSEUDOBLOCK_STOCHASTIC_CG': 4,
     'PSEUDOBLOCK_TFQMR': 5, 'TFQMR': 6, 'LSQR': 7, 'PSEUDOBLOCK_GMRES': 8}).astype(int)
combined['prec_num'] = combined.prec.map({'ILUT': 0, 'RILUK': 1, 'RELAXATION': 2, 'CHEBYSHEV': 3, 'NONE': 4}).astype(int)
combined['status_num'] = combined.status.map({'error': -1, 'unconverged': 0, 'converged': 1}).astype(int)

grouped = combined.groupby(['matrix', 'status_num'])
grouped2 = combined.groupby('matrix')

matrix_best_times = grouped['time'].aggregate(np.min)

# print(matrix_best_times["1138_bus.mtx"][1])

good_bad_series = []
new_time_series = []
for index, row in combined.iterrows(): #iterates through the datframe
    #for mat in matrix_best_times.keys():
    current_matrix_time = row['time']
    matrix_name = row['matrix']
    try:
        matrix_min_time = matrix_best_times[matrix_name][1] # 1 indicates converged
    except:
        matrix_min_time = np.inf
        #print(matrix_name, "exception")

#    if current_matrix_time == matrix_min_time: # check if current time matches best time for that matrix
    if row['status_num'] != 1:
        good_bad_series.append("bad")
        new_time_series.append(np.inf)

        #print(matrix_name, "bad")
    elif current_matrix_time <= 1.25*matrix_min_time:
        good_bad_series.append("good")
        new_time_series.append(current_matrix_time)
        #print(matrix_name, "good")

a = pd.Series(good_bad_series)
a.to_csv('good_bad_list.csv')
b = pd.Series(new_time_series)
b.to_csv('new_time_list.csv')

#print(good_bad_series)
combined = combined.assign(good_or_bad = pd.Series(good_bad_series))
combined.to_csv('hope.csv')
    # todo add in fraction of min
    # todo set error/converged to inf times
    # todo compare only against converged items
    # todo some error checking in case no items converged for that matrix
