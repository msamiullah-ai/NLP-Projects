import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

torch.manual_seed(42)

# ---- Load data ----
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# ---- Build vocabulary ----
counter = Counter()
for text in train_df['text']:
    counter.update(tokenize(text))

vocab_size = 10000
most_common = counter.most_common(vocab_size - 2)  # reserve 0=pad, 1=unk
word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [word2idx.get(tok, 1) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

# ---- Dataset ----
class ReviewDataset(Dataset):
    def __init__(self, df):
        self.texts = [encode(t) for t in df['text']]
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

train_ds = ReviewDataset(train_df)
test_ds = ReviewDataset(test_df)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# ---- Model ----
class TinyNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        avg = embedded.mean(dim=1)             # (batch, embed_dim)
        h = self.relu(self.fc1(avg))
        out = self.fc2(h)
        return out.squeeze(1)

model = TinyNet(vocab_size=len(word2idx))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- Train ----
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

# ---- Evaluate ----
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        out = model(x)
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"\nNeural net test accuracy: {correct/total:.4f}")
