import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class CustomTokenizer:
    
    def __init__(self, stop_words_en):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stop_words_en = stop_words_en
        
    def _lem(self, token):
        if (token in self.stop_words_en):
            return token  # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
        return self.wnl.lemmatize(token)
    
    def __call__(self, doc):
        return [self._lem(t) for t in self.tokenizer.tokenize(doc)]

def stop_words():
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_en.add("b")
    return stop_words_en

def vectorize(train, test, vectorizer):
    features_train = vectorizer.fit_transform(train['headlines'].tolist())
    features_test = vectorizer.transform(test['headlines'].tolist())
    return features_train.todense(), features_test.todense(), vectorizer
    #feature_names = vectorizer.get_feature_names()
    #X_train = pd.DataFrame(features_train.todense(), columns = feature_names)
    #X_test = pd.DataFrame(features_test.todense(), columns = feature_names)
    #return X_train, X_test

def ngram(train, test, n, min_df, max_df):
    # vectorizer = CountVectorizer(tokenizer=CustomTokenizer(stop_words()), stop_words=stop_words(),
                                 # lowercase=True, min_df=min_df,  max_df=max_df, ngram_range=(n, n))
    vectorizer = CountVectorizer(ngram_range=(2,2))
    return vectorize(train, test, vectorizer)

def tfidf(train, test, min_df, max_df):
    # vectorizer = TfidfVectorizer(tokenizer=CustomTokenizer(stop_words()), stop_words=stop_words(),
                                 # lowercase=True, min_df=min_df,  max_df=max_df, ngram_range=(2,2))
    vectorizer = TfidfVectorizer(ngram_range=(2,2), min_df=min_df, max_df=max_df)
    return vectorize(train, test, vectorizer)
