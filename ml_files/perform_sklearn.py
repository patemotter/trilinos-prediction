# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import itertools
import pandas as pd
import numpy as np
import random
from imblearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
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
from sklearn.pipeline import FeatureUnion
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectPercentile
import matplotlib.colors as colors
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, SelectPercentile, SelectKBest, RFECV


class DummySampler(object):
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


def compute_pca(X, y):
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring="f1")
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print(rfecv.scores_)
    print(rfecv.grid_scores_)


def compute_metrics(clf_name, smp_name, y_test, y_pred, file):
    labels = ['-1', '1']
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
    specificity = specificity_score(y_test, y_pred, labels=labels, average=None)
    geo_mean = geometric_mean_score(y_pred, y_pred, labels=labels, average=None)
    iba_gmean = make_index_balanced_accuracy(squared=True)(geometric_mean_score)
    iba = iba_gmean(y_pred, y_test, labels=labels, average=None)
    for i, label in enumerate(labels):
        clf_name = clf_name.strip()
        file.write(clf_name + ',' + smp_name + ',' + str(label) + ',')
        for v in (precision[i], recall[i], specificity[i], f1[i], geo_mean[i],
                  iba[i]):
            file.write(str(v) + ',')
            # print(v, end=',')
        file.write('\n')
        file.flush()
        # print('\n')


def show_roc(clf, clf_name, X_train, y_train, X_test):
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    #  y_score = clf.fit(X_train, y_train).decision_function(X_test)
    print(len(y_score))

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + str(clf_name))
    plt.legend(loc="lower right")
    plt.show()


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
            'Neg Predictive Value: %.2f' % (tn / (tn + fn + 0.)),  # (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Predictive Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()


np.set_printoptions(precision=3)
rng = np.random.RandomState()

# Read files
processed_matrix_properties = pd.read_csv('processed_properties.csv', index_col=0)
processed_timings = pd.read_csv('processed_timings.csv', index_col=0)

# Remove string-based cols
processed_matrix_properties = processed_matrix_properties.drop('matrix', axis=1)
processed_timings = processed_timings.drop('matrix', axis=1)
combined = pd.merge(processed_matrix_properties, processed_timings, on='matrix_id')

# Create training and target sets
X = combined.iloc[:, :-2]
y = combined.iloc[:, -1]

classifier_list = [['GaussianNB', GaussianNB()],
                   ['DecisionTree', DecisionTreeClassifier()],
                   ['LogisticRegression', LogisticRegression()],
                   ['GradientBoosting', GradientBoostingClassifier()],
                   ['KNN', KNeighborsClassifier()]]
samplers_list = [['DummySampler', DummySampler()],
                 ['SMOTE', SMOTE()],
                 ['SMOTEENN', SMOTEENN()],
                 ['SMOTETomek', SMOTETomek()],
                 ['ADASYN', ADASYN()],
                 ['RandomOverSampler', RandomOverSampler()]]

skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X, y)
compute_pca(X, y)

file = open('3_fold_2.txt', 'w')
for clf_name, clf in classifier_list:
    for smp_name, smp in samplers_list:
        for train_index, test_index in skf.split(X, y):
            print(clf_name + ',' + smp_name)  # , smp_name, ',', label, end=',')
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]
            pipeline = pl.make_pipeline(smp, clf)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            # compute_metrics(clf_name, smp_name, y_test, y_pred, file)
            # show_roc(clf, clf_name, X_train, y_train, X_test)
            # print(clf_name, smp_name)
            # file.write(clf_name + ' ' + smp_name)
            # print(classification_report_imbalanced(y_test, y_pred))
            # file.write(classification_report_imbalanced(y_test, y_pred))
file.close()


# plt.show()


# classifier = GradientBoostingClassifier()
# classifier.fit(X_train, y_train)
# y_test_pred = classifier.predict(X_test)
# print("Gradient Boosting Training Score:", classifier.score(X_train, y_train))
# print("Gradient Boosting Test Score:", classifier.score(X_test, y_test))
# print(confusion_matrix(y_test, y_test_pred))
# print(classification_report(y_test, y_test_pred))
# plt.figure()
# cnf_matrix = confusion_matrix(y_test, pipeline.predict(X_test))
# show_confusion_matrix(cnf_matrix, ['-1', '1'])
# plot_confusion_matrix(cnf_matrix, classes=['-1', '1'], normalize=True, title='Normalized confusion matrix')
# plt.matshow(cnf_matrix)
# plt.colorbar()
# plt.show()
