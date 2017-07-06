"""
This script is designed to take already processed matrix timing and properties
files and perform a variety of machine learning techniques using the Scikit-Learn
Python library.
"""  # Written using Anaconda with Python 3.5
# Pate Motter
# 1-22-17

import time
from os import path as path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn import pipeline as pl
from imblearn.over_sampling import RandomOverSampler
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Printing options
np.set_printoptions(precision=3)
rng = np.random.RandomState()

# Number of splits for k-fold cross validation
skf = StratifiedKFold(n_splits=3, random_state=rng)
sss = StratifiedShuffleSplit(n_splits=3, random_state=rng)

stampede = [1, 4, 8, 12, 16]
bridges = [1, 4, 8, 12, 16, 20, 24, 28]
comet = [1, 4, 8, 12, 16, 20, 24]
janus = [1, 2, 4, 6, 8, 10, 12]
summit = [1, 4, 8, 12, 16, 20, 24]
laptop = [1, 2, 4]
JANUS_ID = 0
BRIDGES_ID = 1
COMET_ID = 2
SUMMIT_ID = 3
STAMPEDE_ID = 4
LAPTOP_ID = 5

system_nps = {JANUS_ID: janus, BRIDGES_ID: bridges, COMET_ID: comet,
              SUMMIT_ID: summit, STAMPEDE_ID: stampede, LAPTOP_ID: laptop}

# For roc curves
linestyles = [
    (0, ()),
    (0, (3, 1, 1, 1)),
    (0, (3, 5, 3, 5)),
    (0, (5, 1, 10, 3)),
    (0, (1, 1)),
    (0, (5, 1)),
    (0, (3, 5, 1, 5)),
    (0, (5, 5)),
    (0, (1, 5)),
    (0, (2, 3, 4, 1))
]

classifier_list = [
    ['RandomForest', RandomForestClassifier()]]
    # ['GradientBoosting', GradientBoostingClassifier()],
    # ['GaussianNB', GaussianNB()],
    # ['DecisionTree', DecisionTreeClassifier()],
    # ['LogisticRegression', LogisticRegression()],
    # ['MLP', MLPClassifier()],
    # ['AdaBoost', AdaBoostClassifier()],
    # ['KNN', KNeighborsClassifier()]
    #    ['SVC', SVC(probability=True)],
    #    ['QDA', QuadraticDiscriminantAnalysis()],

samplers_list = [
    ['RandomOverSampler', RandomOverSampler()],
    # ['SMOTE', SMOTE()],
    # ['DummySampler', DummySampler()],
    # ['SMOTEENN', SMOTEENN()],
    # ['SMOTETomek', SMOTETomek()],
    #    ['ADASYN', ADASYN()] prohibitively expensive on larger datasets (3k+ secs)
]


def show_confusion_matrix(C, training_systems, training_numprocs, testing_systems, testing_numprocs,
                          class_labels=['-1', '1']):
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

    confusion_matrix_output = open('cnf_output.csv', 'a')
    # print('TrueNeg\tNumNeg\tTruePos\tNumPos\tFalseNeg\tFalsePos\tTruePosRate\tFalsePosRate\tPosPredVal\tNegPredVal\tAccuracy')
    nps_and_systems = str(testing_systems) + '\t' + str(training_numprocs) + '\t' + \
                      str(testing_systems) + '\t' + str(testing_numprocs)

    cnf_numbers = ('%d\t%d\t%d\t%d\t%d\t%d\t'
                   '%.2f\t%.2f\t%.2f\t%.2f\t%.2f' %
                   (tn, NN, tp, NP, fn, fp,
                    tp / (tp + fn + 0.),
                    fp / (fp + tn + 0.),
                    tp / (tp + fp + 0.),
                    tn / (tn + fn + 0.),
                    (tp + tn + 0.) / N))
    nps_and_systems = nps_and_systems.replace('[', '')
    nps_and_systems = nps_and_systems.replace(']', '')
    confusion_matrix_output.write(nps_and_systems + '\t' + cnf_numbers + '\n')


def classify_good_bad(combined, system, numprocs):
    # process np first
    a = pd.DataFrame()
    if type(numprocs) == str and numprocs == "all":
        a = combined
    elif type(numprocs) == int:
        a = combined[(combined.np == numprocs)]
    elif type(numprocs) == tuple:
        for num in numprocs:
            a = a.append(combined[(combined.numprocs == num)], ignore_index=True)

    # now process systems
    if type(system) == str and system == "all":
        a = a
    elif type(system) == int:
        a = a[(a.system_id == system)]
    elif type(system) == tuple:
        for num in system:
            a = a.append(a[(a.system_id == num)], ignore_index=True)

    # Determine the best times for each matrix
    good_bad_list = []
    new_time_list = []
    grouped = a.groupby(['matrix', 'status_id'])
    best_times = grouped['time'].aggregate(np.min)

    for index, row in a.iterrows():
        current_matrix_time = row['time']
        matrix_name = row['matrix']

        # Check for matrices which never converged
        try:
            matrix_min_time = best_times[matrix_name][1]  # 1 indicates converged
        except:
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
    a.reset_index(drop=True, inplace=True)
    a = a.assign(new_time=pd.Series(new_time_series.values))
    a = a.assign(good_or_bad=pd.Series(good_bad_series))
    return a


def get_properties(properties_filename):
    properties = pd.read_csv(properties_filename, header=0, index_col=0)
    return properties


def get_times(time_files):
    times_array = []
    for t in time_files:
        times_array.append(pd.read_csv(t, header=0, index_col=0))
    combined_times = pd.concat(times_array)
    combined_times = combined_times.drop(labels=['system', 'solver', 'prec', 'status',
                                                 'new_time', 'good_or_bad', 'resid', 'iters'],
                                         axis=1)
    combined_times = combined_times.drop_duplicates()
    return combined_times


def get_classification(combined_times, testing_systems, testing_numprocs):
    start_time = time.time()
    if type(testing_systems) is not list:
        testing_systems = [testing_systems]
    if type(testing_numprocs) is not list:
        testing_numprocs = [testing_numprocs]

    testing_classified = pd.DataFrame()
    for sys in testing_systems:
        for np in testing_numprocs:
            # Check if the np and/or sys even exist
            if np in system_nps[sys]:
                filename = '../classifications/classified_' + str(sys) + '_' + str(np) + '.csv'
                if not path.exists(filename):
                    print("Saving classification to ", filename)
                    temp = classify_good_bad(combined_times, sys, np)
                    testing_classified = testing_classified.append(temp)
                    temp.to_csv(filename)
                    print("Classification time: ", round(time.time() - start_time, 3), '\n')
                else:
                    print('Classification file exists, loading from ' + filename, '\n')
                    temp = pd.read_csv(filename, header=0, index_col=0)
                    testing_classified = testing_classified.append(temp)
    return testing_classified


def merge_properties_and_times(properties_data, timing_data, system_data):
    merged = pd.merge(properties_data, timing_data, on='matrix_id')
    merged = pd.merge(system_data, merged, on='system_id')
    merged = merged.dropna()
    merged = merged.drop(
        labels=['system', 'matrix_y', 'matrix_x', 'status_id', 'time', 'new_time', 'matrix_id'], axis=1)
    return merged


def classify_and_merge(properties_data, timing_data, system_data, specific_nps, specific_systems):
    # Reduce info to just those nps and systems we are wanting to look at
    good_bad_list = []
    new_time_list = []
    specific_nps.sort()
    specific_systems.sort()
    timing_subset = timing_data[timing_data['np'].isin(specific_nps)]
    timing_subset = timing_subset[timing_subset['system_id'].isin(specific_systems)]
    grouped = timing_subset.groupby(['matrix', 'status_id'])
    best_times = grouped['time'].aggregate(np.min)

    filename = '../classifications/classified_' + str(specific_systems) + '_' + str(specific_nps) + '.csv'
    filename = filename.replace(' ', '')
    if path.exists(filename):
        print('Classification file exists, loading from ' + filename, '\n')
        timing_subset = pd.read_csv(filename, header=0, index_col=0)
    else:
        print("Saving classification to ", filename)
        for index, row in timing_subset.iterrows():
            current_matrix_time = row['time']
            matrix_name = row['matrix']

            # Check for matrices which never converged
            try:
                matrix_min_time = best_times[matrix_name][1]  # 1 indicates converged
            except:
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
        timing_subset.reset_index(drop=True, inplace=True)
        timing_subset = timing_subset.assign(new_time=pd.Series(new_time_series.values))
        timing_subset = timing_subset.assign(good_or_bad=pd.Series(good_bad_series))
        timing_subset.to_csv(filename)

    ## Merge the resulting data
    merged = pd.merge(properties_data, timing_subset, on='matrix_id')
    merged = pd.merge(system_data, merged, on='system_id')

    # Remove columns, nans, and duplicates
    merged = merged.dropna()
    merged = merged.drop(
        labels=['system', 'matrix_y', 'matrix_x', 'status_id', 'time', 'new_time', 'matrix_id'], axis=1)
    return merged


def compute_multiple_roc(a, training_systems, training_numprocs, b, testing_systems, testing_numprocs,
                         ls, graph=False):
    total_start_time = time.time()
    output_filename = str(testing_systems) + '_' + str(training_numprocs) + '_' + \
                      str(testing_systems) + '_' + str(testing_numprocs) + '_auroc.csv'
    output_filename = output_filename.replace(' ', '')
    output = open(output_filename, 'w')

    i_a = 0

    # Training and testing data from a
    X_a_train, X_a_test = [], []
    y_a_train, y_a_test = [], []
    X_b_train, X_b_test = [], []
    y_b_train, y_b_test = [], []

    # Set training data to everything but last col, test data is last col
    a_col_list = list(a.columns)
    if a_col_list[-1] != "good_or_bad":
        raise ValueError('Last element in "a" is not good_or_bad')
    X_a = a.iloc[:, :-2]
    y_a = a.iloc[:, -1]

    b_col_list = list(b.columns)
    if b_col_list[-1] != "good_or_bad":
        raise ValueError('Last element in "b" is not good_or_bad')
    X_b = b.iloc[:, :-2]
    y_b = b.iloc[:, -1]

    # Create splits in data using stratified k-fold
    for train_index, test_index in sss.split(X_a, y_a):
        X_a_train.append(X_a.values[train_index])
        X_a_test.append(X_a.values[test_index])
        y_a_train.append(y_a.values[train_index])
        y_a_test.append(y_a.values[test_index])
        i_a += 1

    for train_index, test_index in sss.split(X_b, y_b):
        X_b_train.append(X_b.values[train_index])
        X_b_test.append(X_b.values[test_index])
        y_b_train.append(y_b.values[train_index])
        y_b_test.append(y_b.values[test_index])

    # Permute over the classifiers, samplers, and splits of the data

    best_classifier = ""
    best_sampler = ""
    best_avg = 0.0
    output.write(
        "training_systems\ttraining_numprocs\ttesting_systems\ttesting_numprocs\tclassifier\tsampler\tsplit\tauroc\ttime\n")
    for clf_name, clf in classifier_list:
        for smp_name, smp in samplers_list:
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            total = 0
            pipeline = pl.make_pipeline(smp, clf)

            for split in range(0, i_a):
                start_time = time.time()

                # Fit model to a's training and testing data
                # Compute success of predicting b's testing data using the model
                # cnf = confusion_matrix(y_true=y_test[split], y_pred=pipeline.predict(X_test[split]))
                # show_confusion_matrix(cnf)
                model = pipeline.fit(X_a_train[split], y_a_train[split])
                model_prediction_results = model.predict_proba(X_b_test[split])[:, 1]

                # cnf = confusion_matrix(y_true=y_a_train[split], y_pred=pipeline.predict(X_a_train[split])[:,1]

                test_output = model.predict(X_b_test[split])
                cnf = confusion_matrix(y_b_test[split], test_output)
                show_confusion_matrix(cnf, training_systems, training_numprocs, testing_systems, testing_numprocs)

                # Compute ROC curve and ROC area for each class
                fpr, tpr, _ = roc_curve(y_b_test[split], model_prediction_results)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                wall_time = time.time() - start_time
                # if graph:
                #    plt.plot(fpr, tpr, label='ROC curve - %d (AUC = %0.3f)' % (split, roc_auc))
                total += roc_auc
                print(str(training_systems), str(training_numprocs), str(testing_systems),
                      str(testing_numprocs),
                      clf_name, smp_name,
                      split, round(roc_auc, 3), round(wall_time, 3), sep='\t')
                """
                output.write(
                    str(training_systems) + '\t' + str(training_numprocs) + '\t' + str(
                        testing_systems) + '\t' + str(
                        testing_numprocs) + '\t' +
                    clf_name + '\t' + smp_name + '\t' + str(split) + '\t' + str(round(roc_auc, 3)) +
                    '\t' + str(round(wall_time, 3)) + '\n')
                """

            avg = round(total / float(i_a), 3)
            print(str(training_systems), str(training_numprocs), str(testing_systems),
                  str(testing_numprocs), clf_name,
                  smp_name, "avg", avg,
                  sep='\t', end='')
            output.write(
                str(training_systems) + '\t' + str(training_numprocs) + '\t' + str(
                    testing_systems) + '\t' + str(
                    testing_numprocs) + '\t' +
                clf_name + '\t' + smp_name + '\t' + 'avg' + '\t' + str(avg))

            # Keep track of best results
            if avg > best_avg:
                best_avg = avg
                best_sampler = smp_name
                best_classifier = clf_name

            # Create and save roc graph if desired
            if graph:
                plt.rcParams['lines.linewidth'] = 3
                mean_tpr /= float(i_a)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                training_systems.sort()
                testing_systems.sort()
                plt.plot(mean_fpr, mean_tpr, linestyle=ls,
                         label='{}_{} AUC={:{prec}}'.format(str(training_systems).replace(' ', ''),
                                                            str(testing_systems).replace(' ', ''),
                                                            mean_auc, prec='.2'))
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.legend(loc="lower right", prop={'size': 14})
                plt.tick_params(axis='both', which='major', labelsize=12)
                # plt.savefig('../data/roc_curves/' + str(testing_systems) + '_' + str(
                #    training_numprocs) + '_' + str(
                #    testing_systems) + '_' +
                #            str(testing_numprocs) + '_' + str(clf_name) + '_' +
                #            str(smp_name) + '.svg', bbox_inches='tight')
    print("\t%.2f\n" % (time.time() - total_start_time))


class Experiment:
    def __init__(self, training_sys, training_nps,
                 testing_sys, testing_nps):
        self.training_sys = training_sys
        self.training_nps = training_nps
        self.testing_sys = testing_sys
        self.testing_nps = testing_nps


def createExperiments():
    expList = []
    all_systems = [1, 2, 3, 4, 5]
    all_np = [1, 4, 8, 12, 16, 20, 24, 28]
    i = 0
    expList.append([])

    # For single system, multiple core count experiments
    """
    cur_sys = SUMMIT_ID
    for cur_np in system_nps[cur_sys]:
        expList[i].append(Experiment(training_sys=[cur_sys], training_nps=[1,12,24],
                              testing_sys=[cur_sys], testing_nps=[cur_np]))
    """

    # For fixed core counts, multi-system experiments
    cur_np = 4
    for j in range(1, 6):
        expList.append([])
        expList[i].append(Experiment(training_sys=[j], training_nps=[cur_np],
                                     testing_sys=[BRIDGES_ID], testing_nps=[cur_np]))
        expList[i].append(Experiment(training_sys=[j], training_nps=[cur_np],
                                     testing_sys=[COMET_ID], testing_nps=[cur_np]))
        expList[i].append(Experiment(training_sys=[j], training_nps=[cur_np],
                                     testing_sys=[SUMMIT_ID], testing_nps=[cur_np]))
        expList[i].append(Experiment(training_sys=[j], training_nps=[cur_np],
                                     testing_sys=[STAMPEDE_ID], testing_nps=[cur_np]))
        expList[i].append(Experiment(training_sys=[j], training_nps=[cur_np],
                                     testing_sys=[LAPTOP_ID], testing_nps=[cur_np]))
        i += 1

    return expList


def main():
    # Read in and process properties
    start_time = time.time()
    properties = get_properties('../matrix_properties/processed_properties.csv')

    systems_info = pd.read_csv('../systems_info/systems_info.csv')
    systems_info.system_id = systems_info.system_id.astype(int)

    # Read in and process system timings
    time_files = ['../processed_timings/system_specific/bridges_all_np_timings_processed.csv',
                  '../processed_timings/system_specific/comet_all_np_timings_processed.csv',
                  '../processed_timings/system_specific/laptop_all_np_timings_processed.csv',
                  '../processed_timings/system_specific/summit_all_np_timings_processed.csv',
                  '../processed_timings/system_specific/stampede_all_np_timings_processed.csv']
    combined_times = get_times(time_files)

    # Create training data
    experiments = createExperiments()
    plt.style.use('seaborn-paper')
    for fig in experiments:
        plt.figure()
        ls_iter = 0
        for exp in fig:
            training_classified = get_classification(combined_times, exp.training_sys,
                                                     exp.training_nps)
            training_merged = merge_properties_and_times(properties, training_classified,
                                                         systems_info)

            # Create testing data
            testing_classified = get_classification(combined_times, exp.testing_sys,
                                                    exp.testing_nps)
            testing_merged = merge_properties_and_times(properties, testing_classified,
                                                        systems_info)

            # Compute the prediction ROC
            print(
                "training_systems\t"
                "training_numprocs\t"
                "testing_systems\t"
                "testing_numprocs\t"
                "classifier\t"
                "sampler\t"
                "split\t"
                "auroc\t"
                "time")
            compute_multiple_roc(training_merged, exp.training_sys, exp.training_nps,
                                 testing_merged, exp.testing_sys, exp.testing_nps,
                                 linestyles[ls_iter], graph=True)
            ls_iter += 1
            print("Total execution time: ", round(time.time() - start_time, 3))

    plt.show()


if __name__ == "__main__": main()
