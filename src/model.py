import torch
from torch import nn
import torch.nn.functional as F
import math

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim//2)
        self.fc3 = nn.Linear(embed_dim//2, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        latent = F.dropout((F.relu(self.fc1(embedded))), p=0.4)
        output = F.dropout((F.relu(self.fc2(latent))), p=0.4)
        output = self.fc3(output)
        return output.squeeze()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, vocab_size=20000, d_model=512, nhead=8, num_layers=6, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_layer = nn.Linear(d_model, 1)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        return self.cls_layer(output).squeeze()


class LSTMModel(nn.Module):
    
    def __init__(self, vocab_size=20000, d_model=512, num_layers=6, bidirectional=True, dropout=0.2):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)
        self.cls_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.embedding(x)
        feature = self.lstm(emb).mean(dim=1)
        return self.cls_layer(feature).squeeze()


class CNNModel(nn.Module):

    def __init__(self, vocab_size=20000, d_model=512, kernel_size=5, num_layers=4, dropout=0.2):
        self.embedding = nn.Embedding(vocab_size, d_model)
        modules = []
        for _ in range(num_layers):
            modules.append(nn.Conv1d(d_model, d_model, kernel_size))
            modules.append(nn.BatchNorm1d(d_model))
            modules.append(nn.ReLU)
        self.cnn = nn.Sequential(*modules)
        self.cls_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        feature = self.cnn(emb).mean(dim=-1)
        return self.cls_layer(feature).squeeze()
