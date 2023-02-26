import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

def svm():
    dataframe = pd.read_pickle('add a pickle or csv file)
    print("Read Dataframe from Pickle")
    last_index = (len(dataframe.columns))
    print("Got the length of the dataframe columns")
    binary_index = dataframe.columns.get_loc("Positive sentiment calculated by VADER")
    # dataframemain = dataframe.iloc[:,:]
    dataframemain = dataframe.iloc[:,:binary_index]
    print("Got the records picked")
    clf = OneClassSVM(kernel='rbf', gamma=0.1, nu = 0.1).fit(dataframemain)
    print("Fitted the data to SVM")
    detect_binary_outliers = clf.predict(dataframemain)
    detect_score_outliers = clf.decision_function(dataframemain)
    outliers_only = dataframemain[clf.predict(dataframemain)== -1]
    positives_only = dataframemain[clf.predict(dataframemain) == 1]
    outliers = pd.DataFrame(outliers_only)
    outliers.to_csv(r'abnormalreddits.csv')
    dataframemain["One Class SVM Binary Output"] = detect_binary_outliers
    dataframemain["One Class SVM Score Output"] = np.round(detect_score_outliers,2)
    dataframemain = dataframemain.sort_values(by=["One Class SVM Score Output"])
    dataframemain.to_csv(r'reddit_nodes_svm_output.csv')
    dataframemain.to_pickle('new_redditnodes_svmoutput.pkl')
    print(outliers_only)
    print(positives_only)

def svm_dif_nu():
    dataframe = pd.read_pickle('add a pickle or csv file')
    print("Read Dataframe from Pickle")
    print("Got the length of the dataframe columns")
    binary_index = dataframe.columns.get_loc("Positive sentiment calculated by VADER")
    dataframemain = dataframe.iloc[:, :binary_index]
    print("Got the records picked")
    list_positive, list_negative = [], []
    for kernel in ['poly']:
        for i in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5]:
            clf = OneClassSVM(kernel=kernel, nu=i).fit(dataframemain)
            print("Fitted the data to SVM")
            detect_binary_outliers = clf.predict(dataframemain)
            detect_score_outliers = clf.decision_function(dataframemain)
            outliers_only = dataframemain[clf.predict(dataframemain) == -1]
            positives_only = dataframemain[clf.predict(dataframemain) == 1]
            outliers = pd.DataFrame(outliers_only)
            outliers.to_csv(r'abnormalreddits.csv')
            dataframemain["One Class SVM Binary Output"] = detect_binary_outliers
            dataframemain["One Class SVM Score Output"] = np.round(detect_score_outliers, 2)
            dataframemain = dataframemain.sort_values(by=["One Class SVM Score Output"])
            list_positive.append(len(positives_only.index))
            list_negative.append(len(outliers_only.index))
            print(kernel)
            print(list_positive)
            print(list_negative)

class Run_SVM:
    # svm()
    svm_dif_nu()
