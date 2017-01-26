# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import pandas as pd
import numpy as np
import random
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

np.set_printoptions(precision=5)

# Read files
processed_matrix_properties = pd.read_csv('processed_properties.csv', index_col=0)
processed_timings = pd.read_csv('processed_timings.csv', index_col=0)

# Remove string-based cols
processed_matrix_properties = processed_matrix_properties.drop('matrix', axis=1)
processed_timings = processed_timings.drop('matrix', axis=1)
combined = pd.merge(processed_matrix_properties, processed_timings, on='matrix_id')

# Create training and target sets
X = combined.iloc[0:,0:40]
y = combined.iloc[0:,41]
rng = np.random.RandomState()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng, stratify=y, test_size=0.5)
cross_validation = StratifiedKFold(n_splits=3)

# Based on tutorial 04 - Training and Testing Data
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X, y, cv=cross_validation)
y_test_pred = classifier.predict(X_test)
print("KNN Score" ,classifier.score(X_test,y_test))
print("KNN Scores" , scores)
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

classifier = GaussianNB().fit(X_train, y_train)
classifier.fit(X_train, y_train)
scores = cross_val_score(classifier, X, y, cv=cross_validation)
y_test_pred = classifier.predict(X_test)
print("GaussianNB Score" ,classifier.score(X_test,y_test))
print("GaussianNB Scores" , scores)
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print("Logistic Regression Training Score:", classifier.score(X_train, y_train))
print("Logistic Regression Test Score:", classifier.score(X_test, y_test))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print("Decision Tree Training Score:", classifier.score(X_train, y_train))
print("Decision Tree Test Score:", classifier.score(X_test, y_test))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
print("Gradient Boosting Training Score:", classifier.score(X_train, y_train))
print("Gradient Boosting Test Score:", classifier.score(X_test, y_test))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
