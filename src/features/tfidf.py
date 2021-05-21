import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

def tfidf(train, test):
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_en.add("b")
    vectorizer = TfidfVectorizer(tokenizer=CustomTokenizer(stop_words_en), stop_words=stop_words_en,
                                 lowercase=True, min_df=0.0075,  max_df=0.05, ngram_range=(2,2))
    features_train = vectorizer.fit_transform(train['headlines'].tolist())
    features_test = vectorizer.transform(test['headlines'].tolist())
    feature_names = vectorizer.get_feature_names()
    X_train = pd.DataFrame(features_train.todense(), columns = feature_names)
    X_test = pd.DataFrame(features_test.todense(), columns = feature_names)
    return X_train, X_test
