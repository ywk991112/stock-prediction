import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from os.path import isfile, join, basename, normpath, splitext
from pathlib import Path


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

def get_flair_sentiment(sentence, classifier):
    s = flair.data.Sentence(sentence)
    classifier.predict(s)
    total_sentiment = s.labels
    return total_sentiment[0].to_dict()

def get_transformers_sentiment(sentence, classifier):
    return classifier(sentence)[0]

def estimate(prediction, target):
    def match(est_label, target_label):
        est_label = est_label['label']
        if est_label == 'POSITIVE' and target_label == 1.0:
            return 1
        elif est_label == 'NEGATIVE' and target_label == 0.0:
            return 1
        else:
            return 0
    results = Parallel(n_jobs=4)(delayed(match)(est_label, target_label) for est_label, target_label in zip(prediction, target))
    # return sum(results) / len(target)
    return sum(target) / len(target)

    # y_train, y_test = train['Label'].values, test['Label'].values

if __name__ == '__main__':
<<<<<<< HEAD
    parser = argparse.ArgumentParser("Stock Prediction with Text Information")
    parser.add_argument('--data_path', type=str, default='data/Combined_News_DJIA.csv',
                        help='Path to data csv file')
    parser.add_argument('--model', type=str, default='transformers', choices=['transformers', 'flair'],
                        help='Path to data csv file')
    args = parser.parse_args()

    data = read_data(args.data_path)
    sentiments = []
    if args.model == 'flair':
        import flair
        flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
        for s in tqdm(data['texts'].values):
            sentiments.append(get_flair_sentiment(s, flair_sentiment))
    elif args.model == 'transformers':
        from transformers import pipeline
        classifier = pipeline('sentiment-analysis')
        for s in tqdm(data['title'].values):
            sentiments.append(get_transformers_sentiment(s, classifier))
    corpus_name = splitext(basename(args.data_path))[0]
    Path(join('save', corpus_name)).mkdir(parents=True, exist_ok=True)
    save_path = join('save', corpus_name, f'{args.model}.tar')
    pickle.dump(sentiments, open(save_path, 'wb'))
    with open(save_path, 'rb') as f:
        sentiments = pickle.load(f)

    print(estimate(sentiments, data['Label'].values))
    filename = 'data/Combined_News_Ticker.csv'
    data = read_data(filename)
    # sentiments = Parallel(n_jobs=8)(delayed(get_sentiment_score)(s) for s in tqdm(data['texts'].values))
    sentiments = []
    for s in tqdm(data['texts'].values):
        sentiments.append(get_sentiment_score(s))
    import pickle
    pickle.dump(sentiments, open('results/ticker_sentiment.tar', 'wb'))
