import nltk
import pandas as pd
from features import *

def read_data(filename):
    data = pd.read_csv(filename)
    data['headlines'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    nltk.download('wordnet')
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    return train, test

def get_feature(feature_type, filename):
    train, test = read_data(filename)
    if feature_type == '1-gram':
        x_train, x_test = ngram(train, test, 1)
    elif feature_type == '2-gram':
        x_train, x_test = ngram(train, test, 2)
    elif feature_type == 'tfidf':
        x_train, x_test = tfidf(train, test)
    elif feature_type == 'wv':
        x_train, x_test = word2vec(train, test)
    y_train, y_test = train['Label'], test['Label']
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    filename = 'data/Combined_News_DJIA.csv'
    x_train_1gram, x_test_1gram, y_train, y_test = get_feature('1-gram', filename)
    x_train_2gram, x_test_2gram, y_train, y_test = get_feature('2-gram', filename)
    x_train_tfidf, x_test_tfidf, y_train, y_test = get_feature('tfidf', filename)
    x_train_wv, x_test_wv, y_train, y_test = get_feature('wv', filename)
    print(x_train_1gram)
    print(x_train_2gram)
    print(x_train_tfidf)
    print(x_train_wv)
