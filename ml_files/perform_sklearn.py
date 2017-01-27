# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import itertools
import pandas as pd
import numpy as np
import random
from ggplot import *
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import matplotlib.colors as colors


def show_confusion_matrix(C, class_labels=['0', '1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0];
    fp = C[0, 1];
    fn = C[1, 0];
    tp = C[1, 1];

    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f' % (fp / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Predictive Value: %.2f' % (tn/(tn+fn+0.)), #(1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Predictive Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    #bounds = np.array([0, 0.25, 0.50, 0.75, 1.0])
    #norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)
    #plt.colorbar(ticks=np.linspace(0,1,5))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if i == 0 and j == 0 else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

np.set_printoptions(precision=3)

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

"""
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
"""
#pipeline = pl.make_pipeline(SMOTE(), GradientBoostingClassifier())
classifier_list = [GaussianNB(),
                   DecisionTreeClassifier(),
                   LogisticRegression(),
                   GradientBoostingClassifier(),
                   KNeighborsClassifier()]
for clf in classifier_list:
    pipeline = pl.make_pipeline(clf)
    pipeline.fit(X_train, y_train)
    print(pipeline.steps)
    print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

clf = DecisionTreeClassifier()


#classifier = GradientBoostingClassifier()
#classifier.fit(X_train, y_train)
#y_test_pred = classifier.predict(X_test)
#print("Gradient Boosting Training Score:", classifier.score(X_train, y_train))
#print("Gradient Boosting Test Score:", classifier.score(X_test, y_test))
#print(confusion_matrix(y_test, y_test_pred))
#print(classification_report(y_test, y_test_pred))
#plt.figure()
#cnf_matrix = confusion_matrix(y_test, pipeline.predict(X_test))
#show_confusion_matrix(cnf_matrix, ['-1', '1'])
#plot_confusion_matrix(cnf_matrix, classes=['-1', '1'], normalize=True, title='Normalized confusion matrix')
#plt.matshow(cnf_matrix)
#plt.colorbar()
#plt.show()