from torch import nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.do = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        latent = self.do(F.relu(self.fc1(embedded)))
        output = self.fc2(latent)
        return output.squeeze()
