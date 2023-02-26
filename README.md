# RULEAD: Rule-Based Knowledge Discvery via Anomaly Detection in Tabular Data

### This work is produced as a part of a doctoral degree.


This repository contains the development files of RULEAD for the dataset Reddit. While the dataset related to vital records which is used in the experiments of this work cannot be published publicly, we present the source code of our work using the Reddit dataset. You can download the Reddit dataset from the following link:

https://snap.stanford.edu/data/soc-RedditHyperlinks.html

Use the file names "create_dataframe.py" to generate the features using the dataset downloaded from the above link. Thus created feature matrix can be used to learn a one-class SVM which is implemented in "run_svm.py". Output of this step is available in "abnormalreddits.csv". To cluster the records, use the implementation in "k_means_clustering.py". To build decision tree and random forest, use the respectively named Python files.

Please ensure to change the path information of the input and output files according to your execution environment. While I have used ".pkl" (pickle) files to store outputs as referencing was relatively easier than using a CSV, you are able to change this file format to CSV is you wish to do so. In that case, ensure you change read_pickle to read-csv, and to_csv to to_pickle in every place that they are being used.
