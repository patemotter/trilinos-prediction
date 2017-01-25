# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# Read files
processed_matrix_properties = pd.read_csv('processed_properties.csv', index_col=0)
processed_timings = pd.read_csv('processed_timings.csv', index_col=0)

# Remove string-based cols
processed_matrix_properties = processed_matrix_properties.drop('matrix', axis=1)
processed_timings = processed_timings.drop('matrix', axis=1)
combined = pd.merge(processed_matrix_properties, processed_timings, on='matrix_id')

# Begin ML portion
model = ExtraTreesClassifier()
X = combined.iloc[0:,0:40]
y = combined.iloc[0:,41]
model.fit(X, y)
print(model.feature_importances_)
