# Written using Anaconda with Python 3.5
# Pate Motter
# 1-19-17

# Input;
#   Properties files have many columns of features computed using Anamod
#   Properties file used is trilinos-prediction/data/uflorida-features.csv

import pandas as pd
import numpy as np

matrix_properties = pd.read_csv('../data/uflorida-features.csv', header=0)
matrix_properties.columns = ['rows', 'cols', 'min_nnz_row', 'row_var', 'col_var',
                      'diag_var', 'nnz', 'frob_norm', 'symm_frob_norm',
                      'antisymm_frob_norm', 'one_norm', 'inf_norm', 'symm_inf_norm',
                      'antisymm_inf_norm', 'max_nnz_row', 'trace', 'abs_trace',
                      'min_nnz_row', 'avg_nnz_row', 'dummy_rows', 'dummy_rows_kind',
                      'num_value_symm_1', 'nnz_pattern_symm_1', 'num_value_symm_2',
                      'nnz_pattern_symm_2', 'row_diag_dom', 'col_diag_dom', 'diag_avg',
                      'diag_sign', 'diag_nnz', 'lower_bw', 'upper_bw', 'row_log_val_spread',
                      'col_log_val_spread', 'symm', 'matrix']

# Create hash id's for matrix names
hash_dict = {}
matrix_names = matrix_properties['matrix'].unique()
for name in matrix_names:
    hash_dict[name] = hash(name)
hash_list = []
matrix_name_series = matrix_properties['matrix']
for name in matrix_name_series:
    hash_list.append(hash_dict[name])
matrix_id_series = pd.Series(hash_list)
matrix_properties = matrix_properties.assign(matrix_id=pd.Series(matrix_id_series))

# Fixes any issues with data being > float32, will break some sklearn algos :(
for col in matrix_properties:
    if matrix_properties[col].values.dtype == np.float64:
        matrix_properties[col] = matrix_properties[col].astype(np.float32)
        matrix_properties[col] = np.nan_to_num(matrix_properties[col])

matrix_properties.to_csv('processed_properties.csv')
