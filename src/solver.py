import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import time
from dataloader import get_loader
from model import TextClassificationModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class Solver:
    def __init__(self, features):
        self.x_train, self.x_test, self.y_train, self.y_test, self.vectorizers = features

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x, y):
        raise NotImplementedError
    
class Random_Forest_Solver(Solver):
    def __init__(self, features):
        super().__init__(features)
        self.model = RandomForestClassifier(n_estimators=200, criterion='entropy')

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def evaluate(self):
        return self.model.score(self.x_test, self.y_test)


class Logistic_Solver(Solver):
    def __init__(self, features, C=1.0):
        super().__init__(features)
        self.model = LogisticRegression(C=C)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        return self.model.predict(self.x_test)

    def evaluate(self):
        return self.model.score(self.x_test, self.y_test)

class MLP_Solver(Solver):
    def __init__(self, features, n_epoch):
        super().__init__(features)
        self.train_loader = get_loader(self.x_train, self.y_train, 'train')
        self.test_loader = get_loader(self.x_test, self.y_test, 'test')
        self.model = TextClassificationModel(20000, 512).to(device)
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
