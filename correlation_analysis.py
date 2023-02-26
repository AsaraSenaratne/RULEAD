import scipy.stats
import csv
import pandas as pd
import networkx as nx


def pearson_correlation():
    dataframemain = pd.read_pickle('subreddit_node_features.pkl')
    last_index = (len(dataframemain.columns))
    with open("correlation analysis Reddit nodes.csv", mode='w') as file_obj:
        cluster_writer = csv.writer(file_obj, delimiter=',')
        cluster_writer.writerow(['Feature 1', 'Feature 2', 'Correlation Coefficient', 'p-value'])
        for item1 in range(0,last_index-1):
            column_name_item1 = dataframemain.columns[item1]
            array1 = dataframemain[column_name_item1].to_list()
            for item2 in range(item1+1, last_index):
                column_name_item2 = dataframemain.columns[item2]
                array2 = dataframemain[column_name_item2].to_list()
                coefficient = scipy.stats.pearsonr(array1, array2)
                print(dataframemain.columns[item1],"-", dataframemain.columns[item2], "-", coefficient)
                cluster_writer.writerow([dataframemain.columns[item1], dataframemain.columns[item2], round(coefficient[0],3), round(coefficient[1],3)])

def find_range():
    dataframemain = pd.read_pickle('subreddit_node_features.pkl')
    last_index = (len(dataframemain.columns))
    with open("min max values Reddit nodes.csv", mode='w') as file_obj:
        cluster_writer = csv.writer(file_obj, delimiter=',')
        cluster_writer.writerow(['Feature', 'Min Value', 'Max Value'])
        for item in range(0, last_index - 1):
            column_name_item = dataframemain.columns[item]
            array = dataframemain[column_name_item].to_list()
            min_val = min(array)
            max_val = max(array)
            cluster_writer.writerow([dataframemain.columns[item], min_val, max_val])
            print(dataframemain.columns[item],len(set(array)))

def feature_graph_duplicated():
    dataframe = pd.read_csv("correlation analysis Reddit nodes.csv")
    G = nx.Graph()
    for index, row in dataframe.iterrows():
        if row['Correlation Coefficient'] >= 0.9:
            G.add_edge(row['Feature 1'], row['Feature 2'], weight = row['Correlation Coefficient'])

    cliques = nx.enumerate_all_cliques(G)
    # cliques = nx.find_cliques(G)
    for clique in cliques:
        if len(clique) >= 3:
            weights, count= [], 0
            for i in range(0,len(clique)-1):
                for j in range(i+1,len(clique)):
                    weight = G.get_edge_data(clique[i],clique[j])
                    weights.append(list(weight.values())[0])
            print(clique, weights)

class Correlation_Analysis:
    # pearson_correlation()
    find_range()