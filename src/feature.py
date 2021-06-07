import pandas as pd
from sklearn.model_selection import train_test_split
from features import *

# def read_data(filename):
    # data = pd.read_csv(filename)
    # data['texts'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
    # train = data[data['Date'] < '2015-01-01']
    # test = data[data['Date'] > '2014-12-31']
    # return train, test

def read_data(filename):
    data = pd.read_csv(filename)
    data['texts'] = data['content']
    return train_test_split(data, random_state=123) 

def get_feature(filename, feature_type, param):
    train, test = read_data(filename)
    vectorizer = None
    if feature_type == 'ngram':
        x_train, x_test, vectorizer = ngram(train, test, param['n'], param['min_df'], param['max_df'])
        # x_train, x_test, vectorizer = ngram(train, test, 2, 0.0075, 0.05)
    elif feature_type == 'tfidf':
        x_train, x_test, vectorizer = tfidf(train, test, param['n'], param['min_df'], param['max_df'],
                                            param['max_features'])
    elif feature_type == 'wv':
        x_train, x_test, vectorizer = word2vec(train, test, param['vocab_size'], param['max_length'])
    y_train, y_test = train['Label'].values, test['Label'].values
    return x_train, x_test, y_train, y_test, vectorizer


if __name__ == '__main__':
    filename = 'data/Combined_News_DJIA.csv'
    x_train_1gram, x_test_1gram, y_train, y_test, vectorizer = get_feature('1-gram', filename)
    x_train_2gram, x_test_2gram, y_train, y_test, vectorizer = get_feature('2-gram', filename)
    x_train_tfidf, x_test_tfidf, y_train, y_test, vectorizer = get_feature('tfidf', filename)
    x_train_wv, x_test_wv, y_train, y_test, _ = get_feature('wv', filename)
    # print(x_train_1gram)
    print(x_train_wv)
    # print(x_train_2gram)
    # print(x_train_tfidf)
    # print(x_train_wv)
