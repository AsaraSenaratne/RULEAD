import csv
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image


def automatic_feature_selection_5graphs():
#generate a graph for each dataset size 500,1000,2000,4000,6000,7046
    df = pd.read_pickle('reddit-svmoutput.pkl')
    records = 3523
    # selecting svm binary output and abnormality score columns

    index_of_svm_binary_output = df.columns.get_loc("One Class SVM Binary Output")
    index_of_svm_score_output = df.columns.get_loc("One Class SVM Score Output")

    Y1 = df.iloc[0:records, [index_of_svm_binary_output, index_of_svm_score_output]]
    Y2 = df.iloc[-records:, [index_of_svm_binary_output, index_of_svm_score_output]]
    svm_score = Y1.append(Y2)

    # selecting features
    d_gini_3 = ["LIWC_Verbs","Number of unique works","LIWC_Space"]
    d_3_gini = [df.columns.get_loc(col) for col in d_gini_3]
    d_entropy_3 = ["LIWC_Verbs","LIWC_Body","Fraction of white spaces"]
    d_3_entropy = [df.columns.get_loc(col) for col in d_entropy_3]

    d_gini_6 = ["LIWC_Body","LIWC_Pronoun","LIWC_I",
                "Number of unique works","LIWC_Inhib","LIWC_Numbers"]
    d_6_gini = [df.columns.get_loc(col) for col in d_gini_6]
    d_entropy_6 = ["LIWC_Funct","LIWC_Body","LIWC_Bio","Fraction of white spaces","LIWC_Cause",
                   "Compound sentiment calculated by VADER"]
    d_6_entropy = [df.columns.get_loc(col) for col in d_entropy_6]

    d_gini_9 = ["LIWC_Verbs","Number of unique works","LIWC_Space",
                "LIWC_Prep","LIWC_Bio","Fraction of white spaces","Number of sentences","LIWC_Work"]
    d_9_gini = [df.columns.get_loc(col) for col in d_gini_9]
    d_entropy_9 = ["LIWC_Verbs","LIWC_Body","Fraction of white spaces","Number of unique stopwords","Number of sentences",
                   "Number of words","LIWC_Family","LIWC_Prep"]
    d_9_entropy = [df.columns.get_loc(col) for col in d_entropy_9]

    r_gini_3 = ["Average word length","Number of words","Number of unique stopwords"]
    r_3_gini = [df.columns.get_loc(col) for col in r_gini_3]
    r_entropy_3 = ["Average word length","Number of words","Number of unique stopwords"]
    r_3_entropy = [df.columns.get_loc(col) for col in r_entropy_3]

    r_gini_6 = ["Average word length","Number of words","Number of unique stopwords","Number of unique works",
                "LIWC_Body","Fraction of white spaces"]
    r_6_gini = [df.columns.get_loc(col) for col in r_gini_6]
    r_entropy_6 = ["Average word length","Number of words","Number of unique stopwords","Number of unique works",
                "LIWC_Body","Fraction of white spaces"]
    r_6_entropy = [df.columns.get_loc(col) for col in r_entropy_6]

    r_gini_9 = ["Number of words","LIWC_Body","Average word length","Number of unique works",
                "Number of unique stopwords","Fraction of white spaces","Number of long words","LIWC_Funct"]
    r_9_gini = [df.columns.get_loc(col) for col in r_gini_9]
    r_entropy_9 =  ["Number of words","LIWC_Body","Number of unique works","Average word length",
                "Number of unique stopwords","Fraction of white spaces","Number of long words","LIWC_Funct"]
    r_9_entropy = [df.columns.get_loc(col) for col in r_entropy_9]

    # selecting records
    # Decision Tree
    X1 = df.iloc[0:records, d_3_gini]
    X2 = df.iloc[-records:, d_3_gini]
    d_data_3_gini = X1.append(X2)

    X1 = df.iloc[0:records, d_3_entropy]
    X2 = df.iloc[-records:, d_3_entropy]
    d_data_3_entropy = X1.append(X2)

    X1 = df.iloc[0:records, d_6_gini]
    X2 = df.iloc[-records:, d_6_gini]
    d_data_6_gini = X1.append(X2)

    X1 = df.iloc[0:records, d_6_entropy]
    X2 = df.iloc[-records:, d_6_entropy]
    d_data_6_entropy = X1.append(X2)

    X1 = df.iloc[0:records, d_9_gini]
    X2 = df.iloc[-records:, d_9_gini]
    d_data_9_gini = X1.append(X2)

    X1 = df.iloc[0:records, d_9_entropy]
    X2 = df.iloc[-records:, d_9_entropy]
    d_data_9_entropy = X1.append(X2)

    # Random Forest
    X1 = df.iloc[0:records, r_3_gini]
    X2 = df.iloc[-records:, r_3_gini]
    r_data_3_gini = X1.append(X2)

    X1 = df.iloc[0:records, r_3_entropy]
    X2 = df.iloc[-records:, r_3_entropy]
    r_data_3_entropy = X1.append(X2)

    X1 = df.iloc[0:records, r_6_gini]
    X2 = df.iloc[-records:, r_6_gini]
    r_data_6_gini = X1.append(X2)

    X1 = df.iloc[0:records, r_6_entropy]
    X2 = df.iloc[-records:, r_6_entropy]
    r_data_6_entropy = X1.append(X2)

    X1 = df.iloc[0:records, r_9_gini]
    X2 = df.iloc[-records:, r_9_gini]
    r_data_9_gini = X1.append(X2)

    X1 = df.iloc[0:records, r_9_entropy]
    X2 = df.iloc[-records:, r_9_entropy]
    r_data_9_entropy = X1.append(X2)

    # defining the range of clusters
    K = range(2, 51)
    d_list3_gini, d_list3_entropy = [], []
    d_list6_gini, d_list6_entropy = [], []
    d_list9_gini, d_list9_entropy = [], []
    r_list3_gini, r_list3_entropy = [], []
    r_list6_gini, r_list6_entropy = [], []
    r_list9_gini, r_list9_entropy = [], []

    for k in K:
        # 3 features
        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_3_gini)  # col index 56
        labels = kmeans.labels_
        d_data_3_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_3_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list3_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_3_entropy)  # col index 56
        labels = kmeans.labels_
        d_data_3_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_3_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list3_entropy.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_3_gini)  # col index 56
        labels = kmeans.labels_
        r_data_3_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_3_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list3_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_3_entropy)  # col index 56
        labels = kmeans.labels_
        r_data_3_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_3_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list3_entropy.append(homogeneity)

        # 6 features
        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_6_gini)  # col index 56
        labels = kmeans.labels_
        d_data_6_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_6_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list6_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_6_entropy)  # col index 56
        labels = kmeans.labels_
        d_data_6_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_6_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list6_entropy.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_6_gini)  # col index 56
        labels = kmeans.labels_
        r_data_6_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_6_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list6_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_6_entropy)  # col index 56
        labels = kmeans.labels_
        r_data_6_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_6_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list6_entropy.append(homogeneity)

        # 9 features
        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_9_gini)  # col index 56
        labels = kmeans.labels_
        d_data_9_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_9_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list9_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(d_data_9_entropy)  # col index 56
        labels = kmeans.labels_
        d_data_9_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = d_data_9_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        d_list9_entropy.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_9_gini)  # col index 56
        labels = kmeans.labels_
        r_data_9_gini['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_9_gini['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list9_gini.append(homogeneity)

        kmeans = KMeans(n_clusters=k, random_state=0).fit(r_data_9_entropy)  # col index 56
        labels = kmeans.labels_
        r_data_9_entropy['k-means-cluster'] = labels
        labels_true = svm_score['One Class SVM Binary Output'].values.tolist()
        labels_pred = r_data_9_entropy['k-means-cluster'].values.tolist()
        homogeneity = homogeneity_score(labels_true, labels_pred)
        r_list9_entropy.append(homogeneity)

    # plot 3 features
    plt.plot(K, d_list3_gini, label='DTree - 3 Features - Gini')
    plt.plot(K, d_list3_entropy, label='DTree -3 Features - Entropy')
    # plt.plot(K, r_list3_gini, label='RForest - 3 Features - Gini')
    # plt.plot(K, r_list3_entropy, label='Rforest - 3 Features - Entropy')

    # plot 6 features
    # plt.plot(K, d_list6_gini, label='DTree - 6 Features - Gini')
    # plt.plot(K, d_list6_entropy, label='DTree - 6 Features - Entropy')
    # plt.plot(K, r_list6_gini, label='RForest - 6 Features - Gini')
    # plt.plot(K, r_list6_entropy, label='RForest - 6 Features - Entropy')

    # plot 9 features
    plt.plot(K, d_list9_gini, label='DTree - 9 Features - Gini')
    plt.plot(K, d_list9_entropy, label='DTree - 9 Features - Entropy')
    plt.plot(K, r_list9_gini, label='RForest - 9 Features - Gini')
    plt.plot(K, r_list9_entropy, label='RForest - 9 Features - Entropy')

    plt.xlabel('K')
    plt.ylabel('Homogeneity Score')
    plt.xlim(2, 50)
    plt.ylim(0.05, 1.05)
    plt.title('7046 Records')
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size': 5})
    plt.show()

def assess_cluster_nature():
#Find the cluster of each record, determine the cluster ratio and write the results to a csv file
    df = pd.read_pickle('reddit-svmoutput.pkl')
    records = 3523

    # selecting svm binary output and abnormality score columns

    index_of_svm_binary_output = df.columns.get_loc("One Class SVM Binary Output")
    index_of_svm_score_output = df.columns.get_loc("One Class SVM Score Output")

    Y1 = df.iloc[0:records, [index_of_svm_binary_output, index_of_svm_score_output]]
    Y2 = df.iloc[-records:, [index_of_svm_binary_output, index_of_svm_score_output]]
    svm_score = Y1.append(Y2)

    # selecting features
    feature_list = ["LIWC_Verbs","LIWC_Body","Fraction of white spaces","Number of unique stopwords","Number of sentences",
               "Number of words","LIWC_Family","LIWC_Prep"]
    features = [df.columns.get_loc(col) for col in feature_list]

    # selecting records
    X1 = df.iloc[0:records, features]
    X2 = df.iloc[-records:, features]
    feature_matrix = X1.append(X2)

    K = [10,20,30,40]
    with open("cluster ratios.csv", mode='w') as file_obj:
        cluster_writer = csv.writer(file_obj, delimiter=',')
        cluster_writer.writerow(['Number of Clusters', 'Cluster', 'Normal Records Count (N)', 'Abnormal Records Count (A)', 'Cluster Size (C)','Ratio [max(N,A)/C]','Max (N,A)'])
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(feature_matrix)
            labels = kmeans.labels_
            svm_score['k-means-cluster'] = labels

            for cluster in range(0,k):
                cluster_list = []
                for row in svm_score.itertuples():
                    if row[3] == cluster:
                        cluster_list.append(row[1])
                unique_counts = Counter(cluster_list)
                if unique_counts[1]>unique_counts[-1]:
                    largest = unique_counts[1]
                    name = "N"
                else:
                    largest = unique_counts[-1]
                    name = "A"
                cluster_size = unique_counts[1] + unique_counts[-1]
                try:
                    cluster_writer.writerow([k,cluster,unique_counts[1],unique_counts[-1], cluster_size,round(largest / cluster_size, 2),name])
                except ZeroDivisionError:
                    cluster_writer.writerow([k,cluster,unique_counts[1],unique_counts[-1], cluster_size,"-",name])

def generate_dtree_for_cluster():
    df = pd.read_pickle('svmoutput_abnormalbirths.pkl')
    records = 3523

    # selecting svm binary output and abnormality score columns
    index_of_features_only = len(df.columns)
    index_of_svm_binary_output = df.columns.get_loc("One Class SVM Binary Output")
    index_of_svm_score_output = df.columns.get_loc("One Class SVM Score Output")

    Y1 = df.iloc[0:records, 13:index_of_features_only]
    Y2 = df.iloc[-records:, 13:index_of_features_only]
    svm_score = Y1.append(Y2)

    feature_list = ["Mother's Maiden Name % in Population","Father's Last Name % in Parish","Child's last name matches mother's maiden name",
            "Child's DoB - Parent's DoM","First Name Presence","Common Parish in Population","Common DoB in Population"]
    features = [df.columns.get_loc(col) for col in feature_list]

    # selecting records
    X1 = df.iloc[0:records, features]
    X2 = df.iloc[-records:, features]
    feature_matrix = X1.append(X2)

    K = [20]
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(feature_matrix)
        labels = kmeans.labels_
        svm_score['k-means-cluster'] = labels

    pure_abnormal_cluster = svm_score[svm_score['k-means-cluster']==5]
    pure_abnormal_cluster =  pure_abnormal_cluster.drop(columns=['k-means-cluster'])
    all_negative = pure_abnormal_cluster.iloc[:, 0:index_of_features_only]

    all_the_records = df.iloc[:, 13:index_of_features_only]

    #get the differenc of two dataframes to display rows not common to both
    dataframe_difference = pd.concat([all_negative,all_the_records]).drop_duplicates(keep=False)


    #Draw the tree using all negative records vs all positive records
    # index_of_features_only_1 = len(all_negative.columns) - 2
    # abnormal_cluster = all_negative.iloc[:, 0:index_of_features_only_1]
    # pure_normal_cluster = df[df['One Class SVM Binary Output']==1]
    # pure_normal_cluster = pure_normal_cluster.iloc[:, 13:index_of_features_only-2]
    # X_train = abnormal_cluster.append(pure_normal_cluster)
    # print(X_train)
    #
    # index_of_svm_binary_output1 = df.columns.get_loc("One Class SVM Binary Output")
    # index_of_svm_binary_output2 = all_negative.columns.get_loc("One Class SVM Binary Output")
    # pure_normal_cluster = df[df['One Class SVM Binary Output']==1]
    # all_positive_test = pure_normal_cluster.iloc[:, index_of_svm_binary_output1]
    # all_negative_test = all_negative.iloc[:, index_of_svm_binary_output2]
    # y_train = all_negative_test.append(all_positive_test)
    # print(y_train)

    #Draw tree using all negative vs all the other records
    index_of_features_only_1 = len(all_negative.columns)-2
    abnormal_cluster = all_negative.iloc[:, 0:index_of_features_only_1]
    all_records = dataframe_difference.iloc[:, 0:index_of_features_only_1]
    X_train = abnormal_cluster.append(all_records)

    index_of_svm_binary_output1 = all_negative.columns.get_loc("One Class SVM Binary Output")
    all_negative_test = all_negative.iloc[:, index_of_svm_binary_output1]
    all_positive_test = dataframe_difference.iloc[:, index_of_svm_binary_output1]
    y_train = all_negative_test.append(all_positive_test)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=1, splitter='random')
    clf = clf.fit(X_train, y_train)

    feature_columns = list(df.columns.values)
    features = feature_columns[13:index_of_features_only-2]

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names=features, class_names=['-1', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('DTree-Kmeans-7046-all.png')
    Image(graph.create_png())


def run_classes():
    # automatic_feature_selection_5graphs()
    assess_cluster_nature()
    # generate_dtree_for_cluster()

class KMeans:
    run_classes()