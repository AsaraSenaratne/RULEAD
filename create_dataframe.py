import csv
import operator
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing


def createdataframe():
#this method creates the initial dataframe making baby's the columns and attributes the rows
    with open("soc-redditHyperlinks-body.csv") as file_obj:
        reader = csv.DictReader(file_obj)
        datadict,index = {},0
        for row in reader:
            x = (list(row.values()))
            values = (x[0]+","+x[1][0]+","+x[1][1]+","+x[1][2]+","+x[1][3]+","+x[1][4])
            if len(values) > 77:
                datadict[index] = values.split(",")
                index += 1
        dataframemain = pd.DataFrame(datadict, index=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT', 'Number of characters', "Number of characters without counting white space",
                                                      "Fraction of alphabetical characters", "Fraction of digits", "Fraction of uppercase characters", "Fraction of white spaces", "Fraction of special characters",
                                                      "Number of words","Number of unique works","Number of long words","Average word length","Number of unique stopwords","Fraction of stopwords","Number of sentences",
                                                      "Number of long sentences","Average number of characters per sentence","Average number of words per sentence","Automated readability index","Positive sentiment calculated by VADER",
                                                      "Negative sentiment calculated by VADER","Compound sentiment calculated by VADER","LIWC_Funct","LIWC_Pronoun","LIWC_Ppron","LIWC_I","LIWC_We","LIWC_You","LIWC_SheHe",
                                                      "LIWC_They","LIWC_Ipron","LIWC_Article","LIWC_Verbs","LIWC_AuxVb","LIWC_Past","LIWC_Present","LIWC_Future","LIWC_Adverbs","LIWC_Prep","LIWC_Conj","LIWC_Negate","LIWC_Quant",
                                                      "LIWC_Numbers","LIWC_Swear","LIWC_Social","LIWC_Family","LIWC_Friends","LIWC_Humans","LIWC_Affect","LIWC_Posemo","LIWC_Negemo","LIWC_Anx","LIWC_Anger","LIWC_Sad","LIWC_CogMech","LIWC_Insight",
                                                      "LIWC_Cause","LIWC_Discrep","LIWC_Tentat","LIWC_Certain","LIWC_Inhib","LIWC_Incl","LIWC_Excl","LIWC_Percept","LIWC_See","LIWC_Hear","LIWC_Feel","LIWC_Bio","LIWC_Body",
                                                      "LIWC_Health","LIWC_Sexual","LIWC_Ingest","LIWC_Relativ","LIWC_Motion","LIWC_Space","LIWC_Time","LIWC_Work","LIWC_Achiev","LIWC_Leisure","LIWC_Home","LIWC_Money","LIWC_Relig",
                                                      "LIWC_Death","LIWC_Assent","LIWC_Dissent","LIWC_Nonflu","LIWC_Filler"])
        dataframemain = dataframemain.transpose()
        dataframemain.to_csv(r'reddit-completefeaturefile.csv')
        # pickle_out = open("reddit-dataframe-2.pkl", "wb")
        # pickle.dump(dataframemain, pickle_out, protocol=2)
        # pickle_out.close()
        dataframemain.to_pickle('reddit-dataframe.pkl')
        print(dataframemain)

def get_unique_source_subreddits():
    dataframemain = pd.read_pickle('reddit-dataframe.pkl')
    list_of_source_subredits = dataframemain['SOURCE_SUBREDDIT'].to_list()
    # get unique bundles in the list
    array_of_source_subreddits = np.array(list_of_source_subredits)
    unique_source_subreddits = np.unique(array_of_source_subreddits)
    create_source_subreddit_features(unique_source_subreddits, dataframemain)

def create_source_subreddit_features(unique_source_subreddits, dataframemain):
    dict = {}
    grouped_source_subreddits = dataframemain.groupby(['SOURCE_SUBREDDIT'])
    dataframe_columns = dataframemain.columns.to_series().str.split(',')
    for bundle in unique_source_subreddits:
        feature_list = []
        print(bundle)
        for col in dataframe_columns[5:]:
            one_bundle = grouped_source_subreddits.get_group(bundle)
            one_bundle = one_bundle.astype({col[0]: float})
            avg = (one_bundle[col[0]].sum())/len(one_bundle)
            feature_list.append(round(avg,3))
        dict[bundle] = feature_list
    print("dictionary created")
    with open('source_subreddit_dict.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("dictionary pickled")
    graph_features = pd.DataFrame(dict,index=['Number of characters',
                                         "Number of characters without counting white space",
                                         "Fraction of alphabetical characters", "Fraction of digits",
                                         "Fraction of uppercase characters", "Fraction of white spaces",
                                         "Fraction of special characters",
                                         "Number of words", "Number of unique works", "Number of long words",
                                         "Average word length", "Number of unique stopwords", "Fraction of stopwords",
                                         "Number of sentences",
                                         "Number of long sentences", "Average number of characters per sentence",
                                         "Average number of words per sentence", "Automated readability index",
                                         "Positive sentiment calculated by VADER",
                                         "Negative sentiment calculated by VADER",
                                         "Compound sentiment calculated by VADER", "LIWC_Funct", "LIWC_Pronoun",
                                         "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe",
                                         "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb",
                                         "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep",
                                         "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
                                         "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends",
                                         "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx",
                                         "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight",
                                         "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib",
                                         "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel",
                                         "LIWC_Bio", "LIWC_Body",
                                         "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion",
                                         "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure",
                                         "LIWC_Home", "LIWC_Money", "LIWC_Relig",
                                         "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"])
    graph_features = graph_features.transpose()
    print(graph_features)
    graph_features.to_pickle('source_subreddit_features.pkl')

def get_unique_target_subreddits():
    dataframemain = pd.read_pickle('reddit-dataframe.pkl')
    list_of_target_subredits = dataframemain['TARGET_SUBREDDIT'].to_list()
    # get unique bundles in the list
    array_of_target_subreddits = np.array(list_of_target_subredits)
    unique_target_subreddits = np.unique(array_of_target_subreddits)
    create_target_subreddit_features(unique_target_subreddits, dataframemain)

def create_target_subreddit_features(unique_target_subreddits, dataframemain):
    dict = {}
    grouped_target_subreddits = dataframemain.groupby(['TARGET_SUBREDDIT'])
    dataframe_columns = dataframemain.columns.to_series().str.split(',')
    for bundle in unique_target_subreddits:
        feature_list = []
        print(bundle)
        for col in dataframe_columns[5:]:
            one_bundle = grouped_target_subreddits.get_group(bundle)
            one_bundle = one_bundle.astype({col[0]: float})
            avg = (one_bundle[col[0]].sum())/len(one_bundle)
            feature_list.append(round(avg,3))
        dict[bundle] = feature_list
    print("dictionary created")
    with open('target_subreddit_dict.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("dictionary pickled")
    graph_features = pd.DataFrame(dict, index=['Number of characters',
                                               "Number of characters without counting white space",
                                               "Fraction of alphabetical characters", "Fraction of digits",
                                               "Fraction of uppercase characters", "Fraction of white spaces",
                                               "Fraction of special characters",
                                               "Number of words", "Number of unique works", "Number of long words",
                                               "Average word length", "Number of unique stopwords",
                                               "Fraction of stopwords",
                                               "Number of sentences",
                                               "Number of long sentences", "Average number of characters per sentence",
                                               "Average number of words per sentence", "Automated readability index",
                                               "Positive sentiment calculated by VADER",
                                               "Negative sentiment calculated by VADER",
                                               "Compound sentiment calculated by VADER", "LIWC_Funct", "LIWC_Pronoun",
                                               "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe",
                                               "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb",
                                               "LIWC_Past", "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep",
                                               "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
                                               "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family",
                                               "LIWC_Friends",
                                               "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx",
                                               "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight",
                                               "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain",
                                               "LIWC_Inhib",
                                               "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear",
                                               "LIWC_Feel",
                                               "LIWC_Bio", "LIWC_Body",
                                               "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest", "LIWC_Relativ",
                                               "LIWC_Motion",
                                               "LIWC_Space", "LIWC_Time", "LIWC_Work", "LIWC_Achiev", "LIWC_Leisure",
                                               "LIWC_Home", "LIWC_Money", "LIWC_Relig",
                                               "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu",
                                               "LIWC_Filler"])
    graph_features = graph_features.transpose()
    print(graph_features)
    graph_features.to_pickle('target_subreddit_features.pkl')

def find_source_to_target_pairs():
    dataframemain = pd.read_pickle('reddit-dataframe.pkl')
    list_of_links = []
    for index in range(0, len(dataframemain['SOURCE_SUBREDDIT'])-1):
        source = dataframemain['SOURCE_SUBREDDIT'].iloc[index]
        target = dataframemain['TARGET_SUBREDDIT'].iloc[index]
        list_of_links.append(source + "-" + target)
    array_of_links = np.array(list_of_links)
    unique_links_of_subreddits = np.unique(array_of_links)
    with open('source_to_target_list.pickle', 'wb') as handle:
        pickle.dump(unique_links_of_subreddits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(unique_links_of_subreddits)

def create_dataframe_of_links():
    dict = {}
    source_to_target_list = pd.read_pickle('source_to_target_list.pickle')
    dataframemain_source = pd.read_pickle('source_subreddit_features.pkl')
    dataframemain_target = pd.read_pickle('target_subreddit_features.pkl')
    dataframe_columns = dataframemain_source.columns.to_series().str.split(',')
    for index in range(0, len(source_to_target_list)-1):
        list_of_features = []
        for col in dataframe_columns:
            item = source_to_target_list[index].split("-")
            source = dataframemain_source.loc[item[0], col[0]]
            target = dataframemain_target.loc[item[1], col[0]]
            if source > target:
                list_of_features.append(1)
            else:
                list_of_features.append(0)
        dict[source_to_target_list[index]] = list_of_features
    print(dict)
    graph_features = pd.DataFrame(dict, index=['source_no_of_chars_higher_greater_than_target_no_of_chars',
                                               "source_no_of_chars_without_counting_white_space_greater_than_target_no_of_chars_without_counting_white_space",
                                               "source_fraction_of_alphabetical_chars_greater_than_target_fraction_of_alphabetical_chars",
                                               "source_fraction_of_digits_greater_than_target_fraction_of_digits",
                                               "source_fraction_of_uppercase_chars_greater_than_target_fraction_of_uppercase_chars",
                                               "source_fraction_of_white_spaces_greater_than_target_fraction_of_white_spaces",
                                               "source_fraction_of_special_chars_greater_than_target_fraction_of_special_chars",
                                               "source_no_of_words_greater_than_target_no_of_words",
                                               "source_no_of_unique_works_greater_than_target_no_of_unique_works",
                                               "source_no_of_long_words_greater_than_target_no_of_long_words",
                                               "source_avg_word_length_greater_than_target_avg_word_length",
                                               "source_no_of_unique_stopwords_greater_than_target_no_of_unique_stopwords",
                                               "soource_fraction_of_stopwords_greater_than_target_raction_of_stopwords",
                                               "source_no_of_sentences_greater_than_target_no_of_sentences",
                                               "source_no_of_long_sentences_greater_than_target_no_of_long_sentences",
                                               "source_avg_no_of_chars_per_sentence_greater_than_target_avg_no_of_chars_per_sentence",
                                               "source_avg_no_of_words_per_sentence_greater_than_target_avg_no_of_words_per_sentence",
                                               "source_automated_readability_index_greater_than_target_automated_readability_index",
                                               "source_positive_sentiment_calculated_by_VADER_greater_than_target_positive_sentiment_calculated_by_VADER",
                                               "source_negative_sentiment_calculated_by_VADER_greater_than_target_negative_sentiment_calculated_by_VADER",
                                               "source_compound_sentiment_calculated_by_VADER_greater_than_target_compound_sentiment_calculated_by_VADER",
                                               "source_LIWC_Funct_greater_than_taraget_LIWC_Funct",
                                               "source_LIWC_Pronoun_greater_than_target_LIWC_Pronoun",
                                               "source_LIWC_Ppron_greater_than_target_LIWC_Ppron",
                                               "source_LIWC_I_greater_than_target_LIWC_I",
                                               "source_LIWC_We_greater_than_target_LIWC_We",
                                               "source_LIWC_You_greater_than_target_LIWC_You",
                                               "source_LIWC_SheHe_greater_than_target_LIWC_SheHe",
                                               "source_LIWC_They_greater_than_target_LIWC_They",
                                               "source_LIWC_Ipron_greater_than_target_LIWC_Ipron",
                                               "source_LIWC_Article_greater_than_target_LIWC_Article",
                                               "source_LIWC_Verbs_greater_than_target_LIWC_Verbs",
                                               "source_LIWC_AuxVb_greater_than_target_LIWC_AuxVb",
                                               "source_LIWC_Past_greater_than_target_LIWC_Past",
                                               "source_LIWC_Present_greater_than_target_LIWC_Present",
                                               "source_LIWC_Future_greater_than_target_LIWC_Future",
                                               "source_LIWC_Adverbs_greater_than_target_LIWC_Adverbs",
                                               "source_LIWC_Prep_greater_than_target_LIWC_Prep",
                                               "source_LIWC_Conj_greater_than_target_LIWC_Conj",
                                               "source_LIWC_Negate_greater_than_target_LIWC_Negate",
                                               "source_LIWC_Quant_greater_than_target_LIWC_Quant",
                                               "source_LIWC_Numbers_greater_than_target_LIWC_Numbers",
                                               "source_LIWC_Swear_greater_than_target_LIWC_Swear",
                                               "source_LIWC_Social_greater_than_target_LIWC_Social",
                                               "source_LIWC_Family_greater_than_target_LIWC_Family",
                                               "source_LIWC_Friends_greater_than_target_LIWC_Friends",
                                               "source_LIWC_Humans_greater_than_target_LIWC_Humans",
                                               "source_LIWC_Affect_greater_than_target_LIWC_Affect",
                                               "source_LIWC_Posemo_greater_than_target_LIWC_Posemo",
                                               "source_LIWC_Negemo_greater_than_target_LIWC_Negemo",
                                               "source_LIWC_Anx_greater_than_target_LIWC_Anx",
                                               "source_LIWC_Anger_greater_than_target_LIWC_Anger",
                                               "source_LIWC_Sad_greater_than_target_LIWC_Sad",
                                               "source_LIWC_CogMech_greater_than_target_LIWC_CogMech",
                                               "source_LIWC_Insight_greater_than_target_LIWC_Insight",
                                               "source_LIWC_Cause_greater_than_target_LIWC_Cause",
                                               "source_LIWC_Discrep_greater_than_target_LIWC_Discrep",
                                               "source_LIWC_Tentat_greater_than_target_LIWC_Tentat",
                                               "source_LIWC_Certain_greater_than_target_LIWC_Certain",
                                               "source_LIWC_Inhib_greater_than_target_LIWC_Inhib",
                                               "source_LIWC_Incl_greater_than_target_LIWC_Incl",
                                               "source_LIWC_Excl_greater_than_target_LIWC_Excl",
                                               "source_LIWC_Percept_greater_than_target_LIWC_Percept",
                                               "source_LIWC_See_greater_than_target_LIWC_See",
                                               "source_LIWC_Hear_greater_than_target_LIWC_Hear",
                                               "source_LIWC_Feel_greater_than_target_LIWC_Feel",
                                               "source_LIWC_Bio_greater_than_target_LIWC_Bio",
                                               "source_LIWC_Body_greater_than_target_LIWC_Body",
                                               "source_LIWC_Health_greater_than_target_LIWC_Health",
                                               "source_LIWC_Sexual_greater_than_target_LIWC_Sexual",
                                               "source_LIWC_Ingest_greater_than_target_LIWC_Ingest",
                                               "source_LIWC_Relativ_greater_than_target_LIWC_Relativ",
                                               "source_LIWC_Motion_greater_than_target_LIWC_Motion",
                                               "source_LIWC_Space_greater_than_target_LIWC_Space",
                                               "source_LIWC_Time_greater_than_target_LIWC_Time",
                                               "source_LIWC_Work_greater_than_target_LIWC_Work",
                                               "source_LIWC_Achiev_greater_than_target_LIWC_Achiev",
                                               "source_LIWC_Leisure_greater_than_target_LIWC_Leisure",
                                               "source_LIWC_Home_greater_than_target_LIWC_Home",
                                               "source_LIWC_Money_greater_than_target_LIWC_Money",
                                               "source_LIWC_Relig_greater_than_target_LIWC_Relig",
                                               "source_LIWC_Death_greater_than_target_LIWC_Death",
                                               "source_LIWC_Assent_greater_than_target_LIWC_Assent",
                                               "source_LIWC_Dissent_greater_than_target_LIWC_Dissent",
                                               "source_LIWC_Nonflu_greater_than_target_LIWC_Nonflu",
                                               "source_LIWC_Filler_greater_than_target_LIWC_Filler"])
    graph_features = graph_features.transpose()
    print(graph_features)
    graph_features.to_pickle('subreddit_link_features.pkl')

def analyze_records():
    dict_pairs = {}
    dataframe = pd.read_pickle('reddit-dataframe.pkl')
    dataframemain = dataframe.iloc[:, [0,1]]
    list_of_pairs = dataframemain.values.tolist()
    # print(len(np.unique(x)))
    # print(list_of_pairs[0])
    for item in list_of_pairs:
        key = item[0]+","+item[1]
        if key not in dict_pairs:
            dict_pairs[key] = 1
        else:
            dict_pairs[key] = dict_pairs[key] + 1
    print(sorted(dict_pairs.items(), key=operator.itemgetter(1)))

    dataframemain = dataframe.iloc[:, 0]
    dict_source_reddits, sum = {},0
    list_of_source_reddits = dataframemain.values.tolist()
    for item in list_of_source_reddits:
        if item not in dict_source_reddits:
            dict_source_reddits[item] = 1
        else:
            dict_source_reddits[item] = dict_source_reddits[item] + 1
    print(sorted(dict_source_reddits.items(), key=operator.itemgetter(1)))
    for value in dict_source_reddits.values():
        sum = sum + value
    print("Average posts per group : ", sum/len(dict_source_reddits))

def find_nodes_count():
    dataframemain = pd.read_pickle('reddit-dataframe.pkl')
    list_of_source_subredits = dataframemain['SOURCE_SUBREDDIT'].to_list()
    # get unique bundles in the list
    array_of_source_subreddits = np.array(list_of_source_subredits)
    unique_source_subreddits = np.unique(array_of_source_subreddits)
    list_of_target_subredits = dataframemain['TARGET_SUBREDDIT'].to_list()
    # get unique bundles in the list
    array_of_target_subreddits = np.array(list_of_target_subredits)
    unique_target_subreddits = np.unique(array_of_target_subreddits)
    all_source_target_nodes = np.concatenate((unique_target_subreddits, unique_source_subreddits))
    print(len(np.unique(all_source_target_nodes)))

def merge_source_and_target_reddits():
    source = pd.read_pickle('source_subreddit_features.pkl')
    target = pd.read_pickle('target_subreddit_features.pkl')
    complete_dataframe = pd.concat([source, target]).drop_duplicates()
    normalizedata_nodes(complete_dataframe)

def normalizedata_nodes(complete_dataframe):
#this function perform 0,1 normalization. Accepts the serialized row and returns the normalized row
    dataframe_columns = complete_dataframe.columns.to_series().str.split(',')  # get the columns of the dataframe. Columns represent individual babies
    for col in dataframe_columns:
        row = complete_dataframe[col[0]]
        row_to_normalize = pd.DataFrame(row)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_row = min_max_scaler.fit_transform(row_to_normalize)
        complete_dataframe[col[0]] = np.round(normalized_row, 4)
    print(complete_dataframe)
    complete_dataframe.to_pickle('subreddit_node_features.pkl')

def normalize_links():
    df = pd.read_pickle('subreddit_link_features.pkl')
    dataframe_columns = df.columns.to_series().str.split(',')  # get the columns of the dataframe. Columns represent individual babies
    for col in dataframe_columns:
        row = df[col[0]]
        row_to_normalize = pd.DataFrame(row)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_row = min_max_scaler.fit_transform(row_to_normalize)
        df[col[0]] = np.round(normalized_row, 4)
    print(df)
    df.to_pickle('subreddit_link_features.pkl')

class Create_Dataframe:
    createdataframe()
    analyze_records()
    get_unique_source_subreddits()
    get_unique_target_subreddits()
    find_source_to_target_pairs()
    create_dataframe_of_links()
    find_nodes_count()
    merge_source_and_target_reddits()
    normalize_links()
