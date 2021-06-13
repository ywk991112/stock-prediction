import random
import time
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from model import BertClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BERT_Solver:

    def __init__(self, max_len=500, batch_size=32, epochs=2, seed=123, freeze_bert=False):
        self.max_len = max_len
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.seed = seed
        self.read_data_fn = self.read_data1
        self.freeze_bert = freeze_bert
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def read_data(self, filename):
        data = pd.read_csv(filename)
        data['texts'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
        train = data[data['Date'] < '2015-01-01']
        test = data[data['Date'] > '2014-12-31']
        return train, test

    def read_data1(self, filename):
        data = pd.read_csv(filename)
        data['texts'] = data['content']
        return train_test_split(data, test_size=0.1, random_state=123) 

    def preprocessing_for_bert(self, data):
        input_ids = []
        attention_masks = []

        print('Preprocess data...')
        for sent in tqdm(data):
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  
                add_special_tokens=True,        
                max_length=self.max_len,            
                padding='max_length',
                truncation=True,
                return_attention_mask=True  
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks


    def set_loader(self, X_train, X_val, y_train, y_val):
        print('Tokenizing data...')
        train_inputs, train_masks = self.preprocessing_for_bert(X_train)
        val_inputs, val_masks = self.preprocessing_for_bert(X_val)

        train_labels = torch.tensor(y_train)
        val_labels = torch.tensor(y_val)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        self.val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)


    def initialize_model(self):
        self.model = BertClassifier(self.freeze_bert)
        self.model.to(device)

        self.optimizer = AdamW(self.model.parameters(),
                          lr=5e-5,
                          eps=1e-8
                          )

        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=total_steps)


    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


    def train(self):
        # Start training loop
        print("Start training...\n")
        for epoch_i in range(self.epochs):
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            t0_epoch, t0_batch = time.time(), time.time()

            total_loss, batch_loss, batch_counts = 0, 0, 0

            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch_counts +=1
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                self.model.zero_grad()

                logits = self.model(b_input_ids, b_attn_mask)

                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                if (step % 20 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch

                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            avg_train_loss = total_loss / len(self.train_dataloader)

            print("-"*70)
        
        print("Training complete!")


    def evaluate(self):
        self.model.eval()

        val_accuracy = []
        val_loss = []

        for batch in self.val_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)

            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        time_elapsed = time.time() - t0_epoch
        
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Stock Prediction with Text Information")
    parser.add_argument('--config', type=str, default='bert/config.yaml',
                        help='Config path for the experiment')
    # parser.add_argument('--parallel', type=int, default=1,
                        # help='Run all configs in parallel')
    parser.add_argument('--data_path', type=str, default='data/Combined_News_Ticker_10.csv',
                        help='Path to data csv file')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)
        stream.close()
    solver = BERT_Solver(**config)
    solver.set_seed()
    train, test = solver.read_data_fn(args.data_path)
    solver.set_loader(train['texts'], test['texts'], train['Label'].values, test['Label'].values)
    solver.initialize_model()
    solver.train()
    solver.evaluate()
