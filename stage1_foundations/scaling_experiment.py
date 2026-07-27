import pandas as pd
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import json

torch.manual_seed(42)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

counter = Counter()
for text in train_df['text']:
    counter.update(tokenize(text))

vocab_size = 10000
most_common = counter.most_common(vocab_size - 2)
word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [word2idx.get(tok, 1) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

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

class TinyNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        avg = embedded.mean(dim=1)
        h = self.relu(self.fc1(avg))
        out = self.fc2(h)
        return out.squeeze(1)

def train_and_eval(embed_dim, hidden_dim, epochs=10):
    model = TinyNet(vocab_size=len(word2idx), embed_dim=embed_dim, hidden_dim=hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    n_params = sum(p.numel() for p in model.parameters())
    return avg_loss, accuracy, n_params

configs = [
    {"name": "small",  "embed_dim": 20,  "hidden_dim": 16},
    {"name": "medium", "embed_dim": 50,  "hidden_dim": 64},
    {"name": "large",  "embed_dim": 100, "hidden_dim": 128},
]

results = []
for cfg in configs:
    print(f"\nTraining {cfg['name']} model (embed_dim={cfg['embed_dim']}, hidden_dim={cfg['hidden_dim']})...")
    loss, acc, n_params = train_and_eval(cfg['embed_dim'], cfg['hidden_dim'])
    print(f"{cfg['name']}: test loss={loss:.4f}, accuracy={acc:.4f}, params={n_params}")
    results.append({"name": cfg['name'], "params": n_params, "loss": loss, "accuracy": acc})

with open('results/scaling_results.json', 'w') as f:
    json.dump(results, f, indent=2)

names = [r['name'] for r in results]
params = [r['params'] for r in results]
losses = [r['loss'] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(params, losses, marker='o')
for name, p, l in zip(names, params, losses):
    plt.annotate(name, (p, l), textcoords="offset points", xytext=(5, 5))
plt.xlabel("Number of parameters")
plt.ylabel("Test loss")
plt.title("Mini Scaling Experiment: Model Size vs Test Loss")
plt.grid(True)
plt.savefig('results/scaling_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSummary:")
for r in results:
    print(r)
