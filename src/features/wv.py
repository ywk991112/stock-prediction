import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    headlines_train = train["texts"]
    headlines_test= test["texts"]
    tokens_train = my_tokenizer.tokenize(headlines_train)
    tokens_test = my_tokenizer.tokenize(headlines_test)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train["texts"])

    X_train_tokens = tokenizer.texts_to_sequences(tokens_train)
    X_test_tokens = tokenizer.texts_to_sequences(tokens_test)

    X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

    return X_train_pad, X_test_pad, tokenizer
