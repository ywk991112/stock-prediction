import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

class MyTokenizer():
    def __init__(self):
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words_en = set(nltk.corpus.stopwords.words('english'))
        self.stop_words_germ = set(nltk.corpus.stopwords.words('german'))
        self.stop_words = set()
        self.stop_words.add("b")
        
    def tokenize(self, headlines):
        # Tokenize
        tokens = [self.tokenizer.tokenize(article) for article in headlines]

        # Lemmatizer
        clean_tokens = []
        for words in tokens:
            clean_tokens.append([self.lemmatizer.lemmatize(word) for word in words])

        # Stop words
        final_tokens = []
        for words in clean_tokens:
            final_tokens.append([word.lower() for word in words if word.lower() not in self.stop_words_en \
                                 and word.lower() not in self.stop_words_germ \
                                 and word.lower() not in self.stop_words])
            
        return final_tokens

def word2vec(train, test, vocab_size=20000, max_length=200):
    my_tokenizer = MyTokenizer()

    headlines_train = train["headlines"]
    headlines_test= test["headlines"]
    tokens_train = my_tokenizer.tokenize(headlines_train)
    tokens_test = my_tokenizer.tokenize(headlines_test)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train["headlines"])

    X_train_tokens = tokenizer.texts_to_sequences(tokens_train)
    X_test_tokens = tokenizer.texts_to_sequences(tokens_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

    return X_train_pad, X_test_pad

def read_data(filename):
    data = pd.read_csv(filename)
    data['headlines'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
    nltk.download('stopwords', quiet=True, raise_on_error=True)
    nltk.download('wordnet')
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    return train, test

if __name__ == '__main__':
    filename = 'data/Combined_News_DJIA.csv'
    train, test = read_data(filename)
    x_train_tfidf, x_test_tfidf = tfidf(train, test)
    x_train_wv, x_test_wv = word2vec(train, test)
    print(x_train_tfidf)
    print(x_train_wv)
