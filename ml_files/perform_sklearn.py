# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import pandas as pd
import numpy as np
import preprocessing from sklearn

processed_matrix_properties = pd.read_csv('processed_properties.csv', index_col=0)
processed_timings = pd.read_csv('processed_timings.csv', index_col=0)
processed_matrix_properties = processed_matrix_properties.drop('matrix', axis=1)
processed_timings = processed_timings.drop('matrix', axis=1)

combined = pd.merge(processed_matrix_properties, processed_timings, on='matrix_id')
#print(processed_matrix_properties.head(5))
#print(processed_timings.head(5))
#print(combined.info())

X = combined.iloc[:, 0:40]
Y = combined.iloc[:,41]

normalized_X = preprocessing.normalize(X)