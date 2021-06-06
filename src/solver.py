import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from tqdm import tqdm
import time
from dataloader import get_loader
from model import TextClassificationModel, TransformerModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class Solver:
    def __init__(self, features):
        self.x_train, self.x_test, self.y_train, self.y_test, self.vectorizers = features

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x, y):
        raise NotImplementedError
    
class Sklearn_Solver(Solver):
    def __init__(self, features, model):
        super().__init__(features)
        self.model = model

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def evaluate(self):
        return self.model.score(self.x_test, self.y_test)

    def coeff_words(self, vectorizer):
        basicwords = vectorizer.get_feature_names()
        basiccoeffs = self.model.coef_.tolist()[0]
        coeffdf = pd.DataFrame({'Word' : basicwords,
                                'Coefficient' : basiccoeffs})
        coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        print('Most positive words', coeffdf.head(5))
        print('Most negative words', coeffdf.tail(5))

class Random_Forest_Solver(Sklearn_Solver):
    def __init__(self, features, n_estimators=200, criterion='entropy'):
        super().__init__(features, 
                         RandomForestClassifier(n_estimators=n_estimators, criterion=criterion))

class Logistic_Solver(Sklearn_Solver):
    def __init__(self, features, C=0.8):
        super().__init__(features, LogisticRegression(C=C))

class NB_Solver(Sklearn_Solver):
    def __init__(self, features, alpha=0.01):
        super().__init__(features, MultinomialNB(alpha))
        
class GB_Solver(Sklearn_Solver):
    def __init__(self, features, n_estimators=200):
        super().__init__(features,
                         GradientBoostingClassifier(n_estimators=n_estimators))

class MLP_Solver(Solver):
    def __init__(self, features, model, n_epoch=10):
        super().__init__(features)
        self.train_loader = get_loader(self.x_train, self.y_train, 'train')
        self.test_loader = get_loader(self.x_test, self.y_test, 'test')
        self.model = model.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.n_epoch = n_epoch

    def fit(self):
        self.model.train()
        for epoch in range(1, self.n_epoch + 1):
            epoch_start_time = time.time()
            self.train(epoch)

    def train(self, epoch):
        total_acc, total_count = 0, 0
        log_interval = 50

        for idx, (text, label) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            y_predict = self.model(text)
            loss = self.criterion(y_predict, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            correct_sum, count = self.binary_acc(y_predict, label)
            total_acc += correct_sum.item()
            total_count += count
            if idx % log_interval == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(self.train_loader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0

    @torch.no_grad()
    def evaluate(self):
        total_acc, total_count = 0, 0
        for idx, (text, label) in enumerate(tqdm(self.test_loader)):
            self.optimizer.zero_grad()
            y_predict = self.model(text)
            correct_sum, count = self.binary_acc(y_predict, label)
            total_acc += correct_sum.item()
            total_count += count
        return total_acc/total_count

    @torch.no_grad()
    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        return correct_results_sum, y_test.size(0)

        # acc = correct_results_sum/y_test.shape[0]
        # acc = torch.round(acc * 100)
        # return acc

class FC_Solver(MLP_Solver):
    def __init__(self, features, n_epoch=10, vocab_size=20000, embed_dim=512):
        super().__init__(features, TextClassificationModel(vocab_size, embed_dim), n_epoch)

class TF_Solver(MLP_Solver):
    def __init__(self, features, n_epoch=10, vocab_size=20000, d_model=512,
                 nhead=8, num_layers=6, dropout=0.5):
        super().__init__(features, TransformerModel(vocab_size, d_model, nhead, num_layers, dropout), n_epoch)

def get_solver(solver_type):
    if solver_type == 'logistic':
        return Logistic_Solver
    elif solver_type == 'nb':
        return NB_Solver
    elif solver_type == 'rf':
        return Random_Forest_Solver
    elif solver_type == 'gb':
        return GB_Solver 
    elif solver_type == 'fc':
        return FC_Solver
    elif solver_type == 'tf':
        return TF_Solver
