import flair
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

# def read_data(filename):
    # data = pd.read_csv(filename)
    # data['texts'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
    # train = data[data['Date'] < '2015-01-01']
    # test = data[data['Date'] > '2014-12-31']
    # return train, test

def read_data(filename):
    data = pd.read_csv(filename)
    data['texts'] = data['content']
    return data

def get_sentiment_score(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment[0].to_dict()

# def calculate_accuracy(
    # y_train, y_test = train['Label'].values, test['Label'].values

if __name__ == '__main__':
    filename = 'data/Combined_News_Ticker.csv'
    data = read_data(filename)
    sentiments = Parallel(n_jobs=8)(delayed(get_sentiment_score)(s) for s in tqdm(data['texts'].values))
    import pickle
    pickle.dump(sentiments, open('results/ticker_sentiment.tar', 'wb'))
