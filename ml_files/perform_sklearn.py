"""
This script is designed to take already processed matrix timing and properties
files and perform a variety of machine learning techniques using the Scikit-Learn
Python library.
"""  # Written using Anaconda with Python 3.5
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

    @staticmethod
    def sample(X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


classifier_list = [
    ['GradientBoosting', GradientBoostingClassifier()],
    ['RandomForest', RandomForestClassifier()]]
# ['GaussianNB', GaussianNB()],
# ['DecisionTree', DecisionTreeClassifier()],
# ['LogisticRegression', LogisticRegression()],
# ['MLP', MLPClassifier()],
# ['SVC', SVC(probability=True)],
# ['QDA', QuadraticDiscriminantAnalysis()],
# ['AdaBoost', AdaBoostClassifier()]
# ['KNN', KNeighborsClassifier()]

samplers_list = [
    ['SMOTE', SMOTE()],
    ['RandomOverSampler', RandomOverSampler()]]


# ['DummySampler', DummySampler()],
# ['SMOTEENN', SMOTEENN()],
# ['SMOTETomek', SMOTETomek()],
# ['ADASYN', ADASYN()],


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


def train_and_test(combined, numprocs_a, numprocs_b):
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

    # Create two data frames (a,b) which each contain the datasets for their np number
    a = pd.DataFrame()
    if type(numprocs_a) == str and numprocs_a == "all":
        a = combined[(combined.numprocs != numprocs_a)]
    elif type(numprocs_a) == int:
        a = combined[(combined.numprocs == numprocs_a)]
    elif type(numprocs_a) == list:
        for num in numprocs_a:
            a = a.append(combined[(combined.numprocs == num)], ignore_index=True)
    # Set training data to everything but last col, test data is last col
    X_a = a.iloc[:, :-2]
    y_a = a.iloc[:, -1]
    # Create splits in data using stratified k-fold
    for train_index, test_index in skf.split(X_a, y_a):
        X_a_train.append(X_a.values[train_index])
        X_a_test.append(X_a.values[test_index])
        y_a_train.append(y_a.values[train_index])
        y_a_test.append(y_a.values[test_index])
        i_a += 1

    # Repeat for the second set of data
    b = pd.DataFrame()
    if type(numprocs_b) == str and numprocs_b == "all":
        b = combined[(combined.numprocs != numprocs_b)]
    elif type(numprocs_b) == int:
        b = combined[(combined.numprocs == numprocs_b)]
    elif type(numprocs_b) == list:
        for num in numprocs_b:
            b = b.append(combined[(combined.numprocs == num)], ignore_index=True)
    X_b = b.iloc[:, :-2]
    y_b = b.iloc[:, -1]
    for train_index, test_index in skf.split(X_b, y_b):
        X_b_train.append(X_b.values[train_index])
        X_b_test.append(X_b.values[test_index])
        y_b_train.append(y_b.values[train_index])
        y_b_test.append(y_b.values[test_index])
        i_b += 1

    # Permute over the classifiers, samplers, and splits of the data
    output_file_name = "train_and_test_diff_" + str(numprocs_a) + "_" + str(numprocs_b) + ".csv"
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


def compute_roc(a, numprocs_a, system_a, b, numprocs_b, system_b, output, graph=False):
    """Computes the roc and auc for each split in the two datasets.
    np_a is used as the training data, np_b is used as the testing data"""
    i_a = 0

    # Training and testing data from a
    X_a_train, X_a_test = [], []
    y_a_train, y_a_test = [], []

    # Set training data to everything but last col, test data is last col
    a_col_list = list(a.columns)
    if a_col_list[-1] != "good_or_bad":
        raise ValueError('Last element in "a" is not good_or_bad')
    X_a = a.iloc[:, :-2]
    y_a = a.iloc[:, -1]

    # Create splits in data using stratified k-fold
    for train_index, test_index in skf.split(X_a, y_a):
        X_a_train.append(X_a.values[train_index])
        X_a_test.append(X_a.values[test_index])
        y_a_train.append(y_a.values[train_index])
        y_a_test.append(y_a.values[test_index])
        i_a += 1


    # Training and testing data from b
    X_b_train, X_b_test = [], []
    y_b_train, y_b_test = [], []

    b_col_list = list(b.columns)
    if b_col_list[-1] != "good_or_bad":
        raise ValueError('Last element in "b" is not good_or_bad')
    X_b = b.iloc[:, :-2]
    y_b = b.iloc[:, -1]

    for train_index, test_index in skf.split(X_b, y_b):
        X_b_train.append(X_b.values[train_index])
        X_b_test.append(X_b.values[test_index])
        y_b_train.append(y_b.values[train_index])
        y_b_test.append(y_b.values[test_index])

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

                # Fit model to a's training and testing data
                # Compute success of predicting b's testing data using the model
                y_b_score = pipeline.fit(X_a_train[split], y_a_train[split]).predict_proba(X_b_test[split])[:,1]

                # Compute ROC curve and ROC area for each class
                fpr, tpr, _ = roc_curve(y_b_test[split], y_b_score)
                roc_auc = auc(fpr, tpr)
                wall_time = time.time() - start_time
                if graph:
                    plt.plot(fpr, tpr, label='ROC curve - %d (AUC = %0.3f)'
                                             % (split, roc_auc))
                total += roc_auc
                print(str(system_a), str(numprocs_a), str(system_b), str(numprocs_b), clf_name, smp_name,
                      split, round(roc_auc, 3), round(wall_time, 3), sep='\t')
                output.write(str(system_a) + '\t' + str(numprocs_a) + '\t' + str(system_b) + '\t' + str(numprocs_b) + '\t' +
                             clf_name + '\t' + smp_name + '\t' + str(split) + '\t' + str(round(roc_auc, 3)) +
                             '\t' + str(round(wall_time, 3)) + '\n')

            avg = round(total / float(i_a), 3)
            print(str(system_a), str(numprocs_a), str(system_b), str(numprocs_b), clf_name, smp_name, "avg", avg, sep='\t')
            output.write(str(system_a) + '\t' + str(numprocs_a) + '\t' + str(system_b) + '\t' + str(numprocs_b) + '\t' +
                         clf_name + '\t' + smp_name + '\t' + 'avg' + '\t' + str(avg) + '\n')

            # Keep track of best results
            if avg > best_avg:
                best_avg = avg
                best_sampler = smp_name
                best_classifier = clf_name

            # Create and save roc graph if desired
            if graph:
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve ' + str(clf_name) + " + " + str(smp_name) + "\n" +
                          "Train: " + str(numprocs_a) + "   Test: " + str(numprocs_b))
                plt.legend(loc="lower right")
                plt.savefig('../data/roc_curves/' + str(system_a) + '_' + str(numprocs_a) + '_' + str(system_b) + '_' +
                            str(numprocs_b) + '_' + str(clf_name) + '_' +
                            str(smp_name) + '.svg', bbox_inches='tight')
                plt.close()
    print(str(system_a), str(numprocs_a), str(system_b), str(numprocs_b), best_classifier, best_sampler, "best_avg", best_avg,
          sep='\t')
    output.write(
        str(system_a) + '\t' + str(numprocs_a) + '\t' + str(system_b) + '\t' + str(numprocs_b) + '\t' + best_classifier + '\t' +
        best_sampler + "\tbest_avg\t" + str(best_avg) + '\n')


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

def remove_bad_properties(properties):
    properties = properties.drop(['abs_trace', 'antisymm_frob_norm', 'antisymm_inf_norm',
                                  'col_diag_dom', 'col_log_val_spread', 'col_var',
                                  'cols', 'diag_avg', 'diag_nnz', 'diag_var', 'frob_norm',
                                  'inf_norm', 'min_nnz_row.1', 'nnz_pattern_symm_1',
                                  'nnz_pattern_symm_2', 'one_norm', 'symm', 'symm_frob_norm',
                                  'symm_inf_norm', 'trace'], axis=1)
    return properties


def classify_good_bad(combined, numprocs, system):
    # process np first
    combined.to_csv('test1.csv')
    if type(numprocs) == str and numprocs == "all":
        a = combined
    elif type(numprocs) == int:
        a = combined[(combined.numprocs == numprocs)]
    elif type(numprocs) == list:
        for num in numprocs:
            a = a.append(combined[(combined.numprocs == num)], ignore_index=True)

    # now process systems
    if type(system) == str and system == "all":
        a = a
    elif type(system) == int:
        a = a[(a.system_id == system)]
    elif type(system) == list:
        temp = pd.DataFrame()
        for num in system:
            temp = temp.append(a[(a.system_id == num)], ignore_index=True)
        a = temp

    a.to_csv('test2.csv')
    # Determine the best times for each matrix
    good_bad_list = []
    new_time_list = []
    grouped = a.groupby(['matrix_id', 'status_id'])
    best_times = grouped['time'].aggregate(np.min)

    for index, row in a.iterrows():
        current_matrix_time = row['time']
        matrix_name = row['matrix_id']

        # Check for matrices which never converged

        if matrix_name in best_times:
            matrix_min_time = best_times[matrix_name][1]  # 1 indicates converged
        else:
            matrix_min_time = np.inf

        # Error or unconverged runs = max float time
        if row['status_id'] != 1 or matrix_min_time == np.inf:
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

    # Create Pandas series from the lists which used to contain strings
    new_time_series = pd.Series(new_time_list)
    good_bad_series = pd.Series(good_bad_list)

    # Add the series to the dataframe as columns
    a = a.assign(new_time=pd.Series(new_time_series))
    a = a.assign(good_or_bad=pd.Series(good_bad_series))
    return a


def main():
    # Read in and process properties
    properties_file = '../data/processed_properties.csv'
    properties = pd.read_csv(properties_file, header=0, index_col=0)
    properties = remove_bad_properties(properties)

    # Read in and process system timings
    janus_timefile = '../data/janus/janus_unprocessed_timings.csv'
    bridges_timefile = '../data/bridges/bridges_unprocessed_timings.csv'
    comet_timefile = '../data/comet/comet_unprocessed_timings.csv'

    janus_times = pd.read_csv(janus_timefile, header=0, index_col=0)
    bridges_times = pd.read_csv(bridges_timefile, header=0, index_col=0)
    comet_times = pd.read_csv(comet_timefile, header=0, index_col=0)

    # Concatenate time files together
    times = [janus_times, bridges_times, comet_times]
    combined_times = pd.concat(times)
    #combined_times.columns = ['system_id', 'np', 'matrix_str', 'matrix_id', 'solver_id',
    #                          'prec_id', 'status_id', 'new_time', 'good_or_bad']

    combined_times = combined_times.drop(labels=['system', 'matrix', 'solver', 'prec', 'status',
                                                 'new_time', 'good_or_bad', 'resid'], axis=1)
    combined_times = combined_times.drop_duplicates()

    # Systems: 'janus': 0, 'bridges': 1, 'comet': 2
    # Solvers: FIXED_POINT': 0, 'BICGSTAB': 1, 'MINRES': 2, 'PSEUDOBLOCK_CG': 3, 'PSEUDOBLOCK_STOCHASTIC_CG': 4,
    #    'PSEUDOBLOCK_TFQMR': 5, 'TFQMR': 6, 'LSQR': 7, 'PSEUDOBLOCK_GMRES': 8
    # Preconditioners: 'ILUT': 0, 'RILUK': 1, 'RELAXATION': 2, 'CHEBYSHEV': 3, 'NONE': 4
    # Status: 'error': -1, 'unconverged': 0, 'converged': 1

    # Decide what params to compare to each other
    numprocs_a = 1
    system_a = 0
    numprocs_b = 1
    system_b = 0

    # Create good/bad designation for times depending on the params used
    a = classify_good_bad(combined_times, numprocs_a, system_a)
    b = classify_good_bad(combined_times, numprocs_b, system_b)
    strA = str(numprocs_a) + '_' + str(system_a) + '.csv'
    strB = str(numprocs_b) + '_' + str(system_b) + '.csv'
    a.to_csv(strA)
    b.to_csv(strB)

    # Combine properties and timings
    a = pd.merge(properties, a, on='matrix_id')
    b = pd.merge(properties, b, on='matrix_id')

    # Compute the prediction ROC
    filename = str(system_a) + '_' + str(numprocs_a) + '_' + str(system_b) + '_' + str(numprocs_b) + '_roc-auc.csv'
    output = open(filename, 'w')
    a = b.dropna()
    b = b.dropna()
    a = a.drop(labels=['matrix', 'status_id', 'time', 'new_time', 'matrix_id'], axis=1)
    b = b.drop(labels=['matrix', 'status_id', 'time', 'new_time', 'matrix_id'], axis=1)
    compute_roc(a, numprocs_a, system_a, b, numprocs_b, system_b, output, True)

    #combined = combined.drop(labels=['matrix', 'matrix_str', 'status_id', 'matrix_id'], axis=1)

    #compute_roc(combined, np_a, system_a, np_b, system_b, output, True)

    """
    compute_roc(combined, np_min, np_max,       name, output, True)
    compute_roc(combined, np_min, np_all[1:],   name, output, True)
    compute_roc(combined, np_min, np_all,       name, output, True)
    compute_roc(combined, np_max, np_min,       name, output, True)
    compute_roc(combined, np_max, np_max,       name, output, True)
    compute_roc(combined, np_max, np_all,       name, output, True)
    compute_roc(combined, np_max, np_all[:-1],  name, output, True)
    compute_roc(combined, np_all, np_all,       name, output, True)
    compute_roc(combined, np_all, np_min,       name, output, True)
    compute_roc(combined, np_all, np_max,       name, output, True)
    """
    #output.close()

    # cnf = confusion_matrix(y_true=y_test[split], y_pred=pipeline.predict(X_test[split]))
    # show_confusion_matrix(cnf)


if __name__ == "__main__": main()
