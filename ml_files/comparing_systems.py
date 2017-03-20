import pandas as pd
import numpy as np

janus_timefile = '../data/janus/janus_processed_timings.csv'
bridges_timefile = '../data/bridges/bridges_processed_timings.csv'
comet_timefile = '../data/comet/comet_processed_timings.csv'
properties_file = '../data/processed_properties.csv'

janus_times = pd.read_csv(janus_timefile, header=0, index_col=0)
bridges_times = pd.read_csv(bridges_timefile, header=0, index_col=0)
comet_times = pd.read_csv(comet_timefile, header=0, index_col=0)
properties = pd.read_csv(properties_file, header=0, index_col=0)

janus_times.columns = ['system_id', 'np', 'matrix_str', 'matrix_id', 'solver_id', 
                       'prec_id', 'status_id', 'new_time', 'good_or_bad']
bridges_times.columns = ['system_id', 'np', 'matrix_str', 'matrix_id', 'solver_id', 
                         'prec_id', 'status_id', 'new_time', 'good_or_bad']
comet_times.columns = ['system_id', 'np', 'matrix_str', 'matrix_id', 'solver_id', 
                       'prec_id', 'status_id', 'new_time', 'good_or_bad']

janus_times = janus_times.drop(labels=['matrix_str'], axis=1)
bridges_times = bridges_times.drop(labels=['matrix_str'], axis=1)
comet_times = comet_times.drop(labels=['matrix_str'], axis=1)

janus_times = janus_times.dropna()
bridges_times = bridges_times.dropna()
comet_times = comet_times.dropna()

times = [janus_times, bridges_times, comet_times]
combined_times = pd.concat(times)
combined_times.describe()
properties.describe()

all_combined = pd.merge(properties, combined_times, on='matrix_id')

all_combined.describe()

