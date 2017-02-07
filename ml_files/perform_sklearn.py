"""

This script is designed to take already processed matrix timing and properties
files and perform a variety of machine learning techniques using the Scikit-Learn
Python library.
"""# Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import matplotlib.colors as colors

from imblearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, RandomizedLasso, LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn import over_sampling as os
from sklearn.pipeline import FeatureUnion
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from imblearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, GenericUnivariateSelect, SelectPercentile, SelectKBest, RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Printing options
np.set_printoptions(precision=3)
rng = np.random.RandomState()

# Number of splits for k-fold cross validation
skf = StratifiedKFold(n_splits=3, random_state=rng)


class DummySampler(object):
    """An empty sampler to compare against other classifier and sampler combinations"""
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


classifier_list = [#['GaussianNB', GaussianNB()],
                   #['DecisionTree', DecisionTreeClassifier()],
                   #['LogisticRegression', LogisticRegression()],
                   ['GradientBoosting', GradientBoostingClassifier()],
                   #['MLP', MLPClassifier()],
                   #['SVC', SVC(probability=True)],
                   #['QDA', QuadraticDiscriminantAnalysis()],
                   ['RandomForest', RandomForestClassifier()]]
                   #['AdaBoost', AdaBoostClassifier()]]
                   #['KNN', KNeighborsClassifier()]]

samplers_list = [#['DummySampler', DummySampler()],
                 ['SMOTE', SMOTE()],
                 #['SMOTEENN', SMOTEENN()],
                 #['SMOTETomek', SMOTETomek()],
                 #['ADASYN', ADASYN()],
                 ['RandomOverSampler', RandomOverSampler()]]


def compute_features_rfr(X, y, col_names):
    """Ranks the columns in order of importance based on Random Forest Regression"""
    col_names = col_names[:-1]
    rf = RandomForestRegressor()
    rf.fit(X, y)
    a = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), col_names), reverse=True)
    df = pd.DataFrame(a)
    return df


def compute_features_lasso(X, y, col_names):
    """Ranks the columns in order of importance based on Randomized Lasso (Stability Selection)"""
    col_names = col_names[:-1]
    rl = RandomizedLasso()
    rl.fit(X, y)
    a = sorted(zip(map(lambda x: round(x, 4), rl.scores_), col_names), reverse=True)
    df = pd.DataFrame(a)
    return df


def compute_features_ridge(X, y, col_names):
    """Ranks the columns in order of importance based on Ridge Regression"""
    col_names = col_names[:-1]
    ridge = Ridge()
    ridge.fit(X, y)
    a = sorted(zip(map(lambda x: round(x, 4), ridge.coef_), col_names), reverse=True)
    df = pd.DataFrame(a)
    return df


def compute_features_rfe(X, y, col_names):
    """Ranks the columns in order of importance based on Recursive Feature Elimination (RFE)"""
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, y)
    a = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), col_names))
    df = pd.DataFrame(a)
    return df


def train_and_test(combined, np_a, np_b):
    """
    Trains classifiers on the data in np_a while performing the testing on np_b
    outputs the resulting precision, recall, specificity, f1, geometric mean, and iba
    """
    i_a = 0
    i_b = 0
    X_a_train, X_a_test = [], []
    X_b_train, X_b_test = [], []
    y_a_train, y_a_test = [], []
    y_b_train, y_b_test = [], []

    a = pd.DataFrame()
    if type(np_a) == str and np_a == "all":
        a = combined[(combined.np != np_a)]
    elif type(np_a) == int:
        a = combined[(combined.np == np_a)]
    elif type(np_a) == list:
        for num in np_a:
            a = a.append(combined[(combined.np == num)], ignore_index=True)
    X_a = a.iloc[:, :-2]
    y_a = a.iloc[:, -1]
    for train_index, test_index in skf.split(X_a, y_a):
        X_a_train.append(X_a.values[train_index])
        X_a_test.append(X_a.values[test_index])
        y_a_train.append(y_a.values[train_index])
        y_a_test.append(y_a.values[test_index])
        i_a += 1


    b = pd.DataFrame()
    if type(np_b) == str and np_b == "all":
        b = combined[(combined.np != np_b)]
    elif type(np_b) == int:
        b = combined[(combined.np == np_b)]
    elif type(np_b) == list:
        for num in np_b:
            b = b.append(combined[(combined.np == num)], ignore_index=True)
    X_b = b.iloc[:, :-2]
    y_b = b.iloc[:, -1]
    for train_index, test_index in skf.split(X_b, y_b):
        X_b_train.append(X_b.values[train_index])
        X_b_test.append(X_b.values[test_index])
        y_b_train.append(y_b.values[train_index])
        y_b_test.append(y_b.values[test_index])
        i_b += 1

    # Permute over the classifiers, samplers, and splits of the data
    output_file_name = "train_and_test_diff_" + str(np_a) + "_" + str(np_b) + ".csv"
    output_file = open(output_file_name, 'w')
    header = "classifier,sampler,precision,recall,specificity,f1,geometric_mean,iba"
    print(header)
    output_file.write(header + "\n")
    for clf_name, clf in classifier_list:
        for smp_name, smp in samplers_list:
            pipeline = pl.make_pipeline(smp, clf)
            for split in range(0, i_a):
                pipeline.fit(X_a_train[split], y_a_train[split])
                cur_str = compute_metrics(clf_name, smp_name, y_b_test[split],
                                          pipeline.predict(X_b_test[split]))
                print(cur_str)
                output_file.write(cur_str + "\n")
                output_file.flush()
    output_file.close()


def compute_metrics(clf_name, smp_name, y_test, y_pred):
    """Performs the same tests as Imblearn's classification report.
    Computes precision, recall, specificity, f1, geometric mean, and index balanced accuracy (iba)"""
    labels = ['-1', '1']
    precision, recall, f1, support = \
        precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
    specificity = specificity_score(y_test, y_pred, labels=labels, average=None)
    geo_mean = geometric_mean_score(y_pred, y_pred, labels=labels, average=None)
    iba_gmean = make_index_balanced_accuracy(squared=True)(geometric_mean_score)
    iba = iba_gmean(y_pred, y_test, labels=labels, average=None)
    my_str = ""
    for i, label in enumerate(labels):
        clf_name = clf_name.strip()
        my_str += clf_name + ',' + smp_name + ',' + str(label) + ','
        for v in (precision[i], recall[i], specificity[i], f1[i], geo_mean[i], iba[i]):
            my_str += str(v) + ','
        if i == 0:
            my_str += "\n"
    return my_str


def compute_roc(combined, np_a, np_b, graph=False):
    i_a = 0
    i_b = 0
    X_a_train, X_a_test = [], []
    X_b_train, X_b_test = [], []
    y_a_train, y_a_test = [], []
    y_b_train, y_b_test = [], []

    a = pd.DataFrame()
    if type(np_a) == str and np_a == "all":
        a = combined[(combined.np != np_a)]
    elif type(np_a) == int:
        a = combined[(combined.np == np_a)]
    elif type(np_a) == list:
        for num in np_a:
            a = a.append(combined[(combined.np == num)], ignore_index=True)
    # print(a.info())
    X_a = a.iloc[:, :-2]
    y_a = a.iloc[:, -1]
    for train_index, test_index in skf.split(X_a, y_a):
        X_a_train.append(X_a.values[train_index])
        X_a_test.append(X_a.values[test_index])
        y_a_train.append(y_a.values[train_index])
        y_a_test.append(y_a.values[test_index])
        i_a += 1

    b = pd.DataFrame()
    if type(np_b) == str and np_b == "all":
        b = combined[(combined.np != np_b)]
    elif type(np_b) == int:
        b = combined[(combined.np == np_b)]
    elif type(np_b) == list:
        for num in np_b:
            b = b.append(combined[(combined.np == num)], ignore_index=True)
    # print(b.info())
    X_b = b.iloc[:, :-2]
    y_b = b.iloc[:, -1]
    for train_index, test_index in skf.split(X_b, y_b):
        X_b_train.append(X_b.values[train_index])
        X_b_test.append(X_b.values[test_index])
        y_b_train.append(y_b.values[train_index])
        y_b_test.append(y_b.values[test_index])
        i_b += 1

    # Permute over the classifiers, samplers, and splits of the data
    best_classifier = ""
    best_sampler = ""
    best_avg = 0.0
    for clf_name, clf in classifier_list:
        for smp_name, smp in samplers_list:
            total = 0
            if graph:
                plt.figure()
            pipeline = pl.make_pipeline(smp, clf)
            for split in range(0, i_a):
                start_time = time.time()
                # Fit model to a's training data and attempt to predict b's test data
                y_b_score = pipeline.fit(X_a_train[split],
                                       y_a_train[split]).predict_proba(X_b_test[split])[:,1]
                # Compute ROC curve and ROC area for each class
                fpr, tpr, _ = roc_curve(y_b_test[split], y_b_score)
                roc_auc = auc(fpr, tpr)
                wall_time = time.time() - start_time
                if graph:
                    plt.plot(fpr, tpr, label='ROC curve - %d (AUC = %0.3f)' % (split, roc_auc))
                total += roc_auc
                print(clf_name, smp_name, str(np_a), str(np_b), split, round(roc_auc,3),  round(wall_time,3), sep=',')

            avg = round(total/float(i_a), 3)
            print(clf_name, smp_name, str(np_a), str(np_b), "avg", avg, sep=',')
            if avg > best_avg:
                best_avg = avg
                best_sampler = smp_name
                best_classifier = clf_name
            if graph:
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve ' + str(clf_name) + " + " + str(smp_name) + "\n" +
                          "Train: " + str(np_a) + "   Test: " + str(np_b))
                plt.legend(loc="lower right")
                plt.savefig(str(clf_name) + '_' + str(smp_name) + '_' + str(np_a) + ' ' +
                            str(np_b) + '.svg', bbox_inches='tight')
                plt.close()
    print (best_classifier, best_sampler, np_a, np_b, "best_avg", best_avg, sep=',')

def show_confusion_matrix(C, class_labels=['-1', '1']):
    """Draws confusion matrix with associated metrics"""

    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0]
    fp = C[0, 1]
    fn = C[1, 0]
    tp = C[1, 1]

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


def main():
    # Read files
    processed_matrix_properties = pd.read_csv('processed_properties.csv', index_col=0)
    processed_timings = pd.read_csv('processed_timings.csv', index_col=0)

    # Remove string-based cols
    processed_matrix_properties = processed_matrix_properties.drop('matrix', axis=1)
    processed_timings = processed_timings.drop('matrix', axis=1)
    combined = pd.merge(processed_matrix_properties, processed_timings, on='matrix_id')

    # Dropping details that would not be known via testing
    combined = combined.drop(['new_time', 'matrix_id', 'status_id'], axis=1)

    # Dropping features that were rejected from Pandas-Profile
    combined = combined.drop(['abs_trace', 'antisymm_frob_norm', 'antisymm_inf_norm',
                              'col_diag_dom', 'col_log_val_spread', 'col_var',
                              'cols', 'diag_avg', 'diag_nnz', 'diag_var', 'frob_norm',
                              'inf_norm', 'min_nnz_row.1', 'nnz_pattern_symm_1',
                              'nnz_pattern_symm_2', 'one_norm', 'symm', 'symm_frob_norm',
                              'symm_inf_norm', 'trace'], axis=1)

    combined = combined.drop_duplicates()
    """
    train_and_test(combined, 1, 1)
    train_and_test(combined, 2, 2)
    train_and_test(combined, 4, 4)
    train_and_test(combined, 6, 6)
    train_and_test(combined, 8, 8)
    train_and_test(combined, 10, 10)
    train_and_test(combined, 12, 12)
    """

    all = [1, 2, 4, 6, 8, 10, 12]
    #compute_roc(combined, 1, 1)
    #compute_roc(combined, 1, 12)
    #compute_roc(combined, 12, 1)
    #compute_roc(combined, 12, 12)
    #compute_roc(combined, 1, [2,4,6,8,10,12])
    compute_roc(combined, all, all)
    compute_roc(combined, all, 1)
    compute_roc(combined, all, 12)
    #compute_roc(combined, 1, all)
    #compute_roc(combined, 12, all)

    # print(classification_report_imbalanced(y_test, pipeline.predict(X_test[split])))

    # cnf = confusion_matrix(y_true=y_test[split], y_pred=pipeline.predict(X_test[split]))
    # show_confusion_matrix(cnf)


if __name__ == "__main__": main()
