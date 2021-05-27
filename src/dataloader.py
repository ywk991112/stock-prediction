import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class NewsDataset(Dataset):
    def __init__(self, text_indices, label):
        self.text_indices = text_indices
        self.label = label
    def __len__(self):
        return len(self.text_indices)
    def __getitem__(self, idx):
        return self.text_indices[idx], self.label[idx]

def collate_batch(batch):
    text_list, label_list = [], []
    for (_indices, _label) in batch:
         processed_text = torch.tensor(_indices, dtype=torch.int64)
         text_list.append(processed_text)
         label_list.append(_label)
    text_list = torch.stack(text_list)
    label_list = torch.tensor(label_list).float()
    return text_list.to(device), label_list.to(device)

def get_loader(text_indices, labels, mode):
    if mode == 'train':
        return DataLoader(NewsDataset(text_indices, labels), batch_size=8,
                          shuffle=True, collate_fn=collate_batch)
    elif mode == 'test':
        return DataLoader(NewsDataset(text_indices, labels), batch_size=8,
                          shuffle=False, collate_fn=collate_batch)
