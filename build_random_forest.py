# this file generates a Random Forest under gini and entropy and writes the scores into a .csv file
# the trees can be viewed from RForest-Entropy.png and RForest-Gini.png files
# The output of oneclasssvm.py is used as an input here. The pickle output file

import numpy as np
import pandas as pd
import pydotplus
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image


def randomforest_500():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]


    # first 250 and last 250 records = 500 records
    X1 = dataframemain.iloc[0:250, 5:index_of_features_only]
    X2 = dataframemain.iloc[-250:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:250, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-250:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # To use all records for testing
    # X_train = X3
    # y_train = Y3
    # X_test = dataframemain.iloc[:, 13:53]
    # y_test = dataframemain.iloc[:, 53]

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    # dataframe_features = dataframe_features.sort_values(by=["Entropy"], ascending= False) #Sort the dataframe itself
    dataframe_features.to_csv(r'Importance_of_Features_RForest_500.csv')

    # Get node level information
    # tree_explianed = (tree.plot_tree(clf, label='all',  impurity='True', node_ids='True', class_names=['1','-1'], precision=3))
    # for item in tree_explianed:
    #     print(item)
    # export_tree = export_text(tree_explianed)

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-500.png')
    Image(graph.create_png())

    # # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-500.png')
    Image(graph.create_png())
    randomforest_1000()

def randomforest_1000():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]

    # first 500 and last 500 records = 1000 records
    X1 = dataframemain.iloc[0:500, 5:index_of_features_only]
    X2 = dataframemain.iloc[-500:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:500, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-500:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    dataframe_features.to_csv(r'Importance_of_Features_RForest_1000.csv')

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-1000.png')
    Image(graph.create_png())

    # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-1000.png')
    Image(graph.create_png())
    randomforest_2000()

def randomforest_2000():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]

    # first 1000 and last 1000 records = 2000 records
    X1 = dataframemain.iloc[0:1000, 5:index_of_features_only]
    X2 = dataframemain.iloc[-1000:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:1000, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-1000:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    dataframe_features.to_csv(r'Importance_of_Features_RForest_2000.csv')

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-2000.png')
    Image(graph.create_png())

    # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-2000.png')
    Image(graph.create_png())
    randomforest_4000()

def randomforest_4000():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]

    # first 2000 and last 2000  records = 4000 records
    X1 = dataframemain.iloc[0:2000, 5:index_of_features_only]
    X2 = dataframemain.iloc[-2000:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:2000, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-2000:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    dataframe_features.to_csv(r'Importance_of_Features_RForest_4000.csv')

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-4000.png')
    Image(graph.create_png())

    # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-4000.png')
    Image(graph.create_png())
    randomforest_6000()

def randomforest_6000():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]

    # first 3000 and last 3000 records =  6000 records
    X1 = dataframemain.iloc[0:3000, 5:index_of_features_only]
    X2 = dataframemain.iloc[-3000:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:3000, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-3000:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    dataframe_features.to_csv(r'Importance_of_Features_RForest_6000.csv')

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-6000.png')
    Image(graph.create_png())

    # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-6000.png')
    Image(graph.create_png())
    randomforest_7046()

def randomforest_7046():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('reddit-svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[5:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]

    # first 3523 and last 3523 records =  7046 records
    X1 = dataframemain.iloc[0:3523, 5:index_of_features_only]
    X2 = dataframemain.iloc[-3523:, 5:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:3523, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-3523:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[5:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    dataframe_features.to_csv(r'Importance_of_Features_RForest_7046.csv')

    #Save the decision tree in a .png file
    #Gini
    dot_data = StringIO()
    export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('RForest-Gini-7046.png')
    Image(graph.create_png())

    # Entropy
    dot_data1 = StringIO()
    export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph.write_png('RForest-Entropy-7046.png')
    Image(graph.create_png())

def randomforest():
    #pull out pickled svmoutput
    dataframemain = pd.read_pickle('new_redditnodes_svmoutput.pkl')
    index_of_features_only = len(dataframemain.columns) - 2
    index_of_svm_binary_output = dataframemain.columns.get_loc("One Class SVM Binary Output")

    #Seperate the feature columns and label column
    feature_columns = list(dataframemain.columns.values)
    features = feature_columns[0:index_of_features_only]
    # features = ['First name % in Population', 'Last name % in Population', 'Parish % in Population', 'Gender % in Population', 'Address % in Population', "Father's Occupation % in Population", "Mother's First Name % in Population", "Mother's Maiden Name % in Population", "Mother's Last Name % in Population", "Father's First Name % in Population", "Father's Last Name % in Population", 'DoB % in Population', 'First name % in Parish', 'Last name % in Parish', 'Gender % in Parish', 'Address % in Parish', "Father's Occupation % in Parish", "Mother's First Name % in Parish", "Mother's Maiden Name % in Parish", "Mother's Last Name % in Parish", "Father's First Name % in Parish", "Father's Last Name % in Parish", 'DoB % in Parish', "Child's last name matches father's last name", "Child's last name matches mother's last name", "Child's last name matches mother's maiden name", "Child's DoB - Parent's DoM", 'First Name Presence', 'Last Name Presence', 'Parish Presence', 'Gender Presence', 'DoB Presence', 'Address Presence', "Father's Occupation Presence", "Mother's First Name Presence", "Mother's Maiden Name Presence", "Mother's Last Name Presence", "Father's First Name Presence", "Father's Last Name Presence", "Parents' Marriage Date Presence", 'First name % in Population le 0.35', 'First name % in Population ge 0.36 le 0.66', 'First name % in Population ge 0.67', 'Last name % in Population le 0.35', 'Last name % in Population ge 0.36 le 0.66', 'Last name % in Population ge 0.67', 'Parish % in Population le 0.35', 'Parish % in Population ge 0.36 le 0.66', 'Parish % in Population ge 0.67', 'Gender % in Population le 0.35', 'Gender % in Population ge 0.36 le 0.66', 'Gender % in Population ge 0.67', 'Address % in Population le 0.35', 'Address % in Population ge 0.36 le 0.66', 'Address % in Population ge 0.67', "Father's Occupation % in Population le 0.35", "Father's Occupation % in Population ge 0.36 le 0.66", "Father's Occupation % in Population ge 0.67", "Mother's First Name % in Population le 0.35", "Mother's First Name % in Population ge 0.36 le 0.66", "Mother's First Name % in Population ge 0.67", "Mother's Last Name % in Population le 0.35", "Mother's Last Name % in Population ge 0.36 le 0.66", "Mother's Last Name % in Population ge 0.67", "Father's First Name % in Population le 0.35", "Father's First Name % in Population ge 0.36 le 0.66", "Father's First Name % in Population ge 0.67", "Father's Last Name % in Population le 0.35", "Father's Last Name % in Population ge 0.36 le 0.66", "Father's Last Name % in Population ge 0.67", "DoB % in Population le 0.35", "DoB % in Population ge 0.36 le 0.66", "DoB % in Population ge 0.67", "First name % in Parish le 0.35", "First name % in Parish ge 0.36 le 0.66", "First name % in Parish ge 0.67", "Last name % in Parish le 0.35", "Last name % in Parish ge 0.36 le 0.66", "Last name % in Parish ge 0.67", "Gender % in Parish le 0.35", "Gender % in Parish ge 0.36 le 0.66", "Gender % in Parish gt 0.67", "Address % in Parish le 0.35", "Address % in Parish ge 0.36 le 0.66", "Address % in Parish ge 0.67", "Father's Occupation % in Parish le 0.35", "Father's Occupation % in Parish ge 0.36 le 0.66", "Father's Occupation % in Parish ge 0.67", "Mother's First Name % in Parish ge 0.36 le 0.66", "Mother's First Name % in Parish ge 0.67", "Mother's Last Name % in Parish le 0.35", "Mother's Last Name % in Parish ge 0.36 le 0.66", "Mother's Last Name % in Parish ge 0.67", "Father's First Name % in Parish le 0.35", "Father's First Name % in Parish ge 0.36 le 0.66", "Father's First Name % in Parish ge 0.67", "Father's Last Name % in Parish le 0.35", "Father's Last Name % in Parish ge 0.36 le 0.66", "Father's Last Name % in Parish ge 0.67", "DoB % in Parish le 0.35", "DoB % in Parish ge 0.36 le 0.66", "DoB % in Parish ge 0.67"]


    # first 250 and last 250 records = 500 records
    X1 = dataframemain.iloc[0:4000, 0:index_of_features_only]
    X2 = dataframemain.iloc[-4000:, 0:index_of_features_only]
    X3 = X1.append(X2)
    Y1 = dataframemain.iloc[0:4000, index_of_svm_binary_output]
    Y2 = dataframemain.iloc[-4000:, index_of_svm_binary_output]
    Y3 = Y1.append(Y2)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X3, Y3, test_size = 0.3)  # 70% training and 30% test on selected records

    # To use all records for testing
    # X_train = X3
    # y_train = Y3
    # X_test = dataframemain.iloc[:, 13:53]
    # y_test = dataframemain.iloc[:, 53]

    # Create Decision Tree classifer object
    clf = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='gini')
    clf.compute_importances = True
    clf1 = RandomForestClassifier(max_depth=4, random_state=1, n_estimators=2000, criterion='entropy')
    clf1.compute_importances = True
    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)
    clf1 = clf1.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    estimator1 = clf.estimators_[50]
    print("Accuracy Gini:",metrics.accuracy_score(y_test, y_pred))  # Model Accuracy, how often is the classifier correct?

    y_pred = clf1.predict(X_test)
    estimator2 = clf1.estimators_[50]
    print("Accuracy Entropy:", metrics.accuracy_score(y_test, y_pred))

    #Get the feature importance written to csv
    dataframe_features = dataframemain.columns[0:index_of_features_only]
    dataframe_features = pd.DataFrame(dataframe_features)
    dataframe_features['Gini'] = np.round(clf.feature_importances_,2)
    dataframe_features['Rank - G'] = dataframe_features['Gini'].rank(method='dense', ascending=False) #rank the 'Gini' column and write the output to 'Rank-G' column
    dataframe_features.loc[(dataframe_features['Gini'] == 0) , 'Rank - G'] = "" #Remove the ranks assigned to any gini score with 0
    dataframe_features['Entropy'] = np.round(clf1.feature_importances_, 2)
    dataframe_features['Rank - E'] = dataframe_features['Entropy'].rank(method='dense', ascending=False)
    dataframe_features.loc[(dataframe_features['Entropy'] == 0), 'Rank - E'] = ""
    dataframe_features.columns = ['Features', 'Gini', 'Rank - G','Entropy', 'Rank - E']
    # dataframe_features = dataframe_features.sort_values(by=["Entropy"], ascending= False) #Sort the dataframe itself
    dataframe_features.to_csv(r'Importance_of_Features_RForest_redditnodes.csv')

    # Get node level information
    # tree_explianed = (tree.plot_tree(clf, label='all',  impurity='True', node_ids='True', class_names=['1','-1'], precision=3))
    # for item in tree_explianed:
    #     print(item)
    # export_tree = export_text(tree_explianed)

    #Save the decision tree in a .png file
    #Gini
    # dot_data = StringIO()
    # export_graphviz(estimator1, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['1', '-1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('RForest-Gini-500.png')
    # Image(graph.create_png())
    #
    # # # Entropy
    # dot_data1 = StringIO()
    # export_graphviz(estimator2, out_file=dot_data1, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['1', '-1'])
    # graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    # graph.write_png('RForest-Entropy-500.png')
    # Image(graph.create_png())
    # randomforest_1000()

class RandomForest:
    # randomforest_500()
    randomforest()