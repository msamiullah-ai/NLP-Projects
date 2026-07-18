"""
Vanilla RNN Character-Level Language Model — implemented with raw matrix ops.
No nn.RNN, no nn.RNNCell, no nn.Linear for the recurrent core.
Autograd still computes gradients for us (that part is legitimate to lean on;
what we're proving we understand is the FORWARD PASS equations and the
mechanics of BPTT-through-time unrolling, not reimplementing autograd itself).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
""" * 40  # repeat so the tiny corpus is long enough for a real batch

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
print(f"Corpus length: {len(text)} chars | Vocab size: {vocab_size}")

# ----------------------------------------------------------------------
# 2. BATCHING
# ----------------------------------------------------------------------
seq_len = 25
batch_size = 16

def get_batch():
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y  # (batch, seq_len)

# ----------------------------------------------------------------------
# 3. THE RNN CELL — RAW MATRIX OPS
#
#    h_t = tanh( x_t @ Wxh + h_{t-1} @ Whh + bh )
#    y_t = h_t @ Why + by
#
#    Shapes:
#      x_t : (batch, vocab_size)      one-hot input at time t
#      Wxh : (vocab_size, hidden)
#      Whh : (hidden, hidden)
#      Why : (hidden, vocab_size)
# ----------------------------------------------------------------------
class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Xavier-ish init so training doesn't blow up on step 1
        self.Wxh = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x, h_prev=None):
        """
        x: (batch, seq_len) of token indices
        returns: logits (batch, seq_len, vocab_size), final hidden state
        """
        batch, T = x.shape
        if h_prev is None:
            h_prev = torch.zeros(batch, self.hidden_size, device=x.device)

        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()  # (batch, T, vocab)

        logits = []
        h = h_prev
        for t in range(T):
            x_t = x_onehot[:, t, :]                      # (batch, vocab)
            h = torch.tanh(x_t @ self.Wxh + h @ self.Whh + self.bh)  # (batch, hidden)
            y_t = h @ self.Why + self.by                  # (batch, vocab)
            logits.append(y_t)

        logits = torch.stack(logits, dim=1)  # (batch, T, vocab)
        return logits, h.detach()  # detach so next batch doesn't backprop forever

# ----------------------------------------------------------------------
# 4. TRAINING SETUP
# ----------------------------------------------------------------------
hidden_size = 128
model = VanillaRNNLM(vocab_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

n_steps = 800
loss_history = []

# ----------------------------------------------------------------------
# 5. SAMPLING / GENERATION
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(model, seed_str, length=200, temperature=0.8):
    model.eval()
    h = torch.zeros(1, model.hidden_size)
    idxs = [char_to_idx[c] for c in seed_str]

    # warm up hidden state on the seed string
    for i in idxs[:-1]:
        x_t = F.one_hot(torch.tensor([i]), num_classes=vocab_size).float()
        h = torch.tanh(x_t @ model.Wxh + h @ model.Whh + model.bh)

    current = idxs[-1]
    out_chars = list(seed_str)
    for _ in range(length):
        x_t = F.one_hot(torch.tensor([current]), num_classes=vocab_size).float()
        h = torch.tanh(x_t @ model.Wxh + h @ model.Whh + model.bh)
        logits = h @ model.Why + model.by
        probs = F.softmax(logits / temperature, dim=-1).squeeze()
        current = torch.multinomial(probs, num_samples=1).item()
        out_chars.append(idx_to_char[current])

    model.train()
    return "".join(out_chars)

# ----------------------------------------------------------------------
# 6. TRAINING LOOP
# ----------------------------------------------------------------------
h_state = None
for step in range(n_steps):
    x, y = get_batch()

    logits, h_state = model(x, h_state)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # RNNs need this
    optimizer.step()

    loss_history.append(loss.item())

    if step % 200 == 0 or step == n_steps - 1:
        print(f"step {step:4d} | loss {loss.item():.4f}")
        sample = generate(model, seed_str="First", length=80, temperature=0.7)
        print(f"  sample: {sample!r}\n")

# ----------------------------------------------------------------------
# 7. PLOT LOSS CURVE
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("Vanilla RNN (from scratch) — training loss")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=120)
print("Saved loss_curve.png")

print("\nFinal generation sample:")
print(generate(model, seed_str="First Citizen:", length=300, temperature=0.8))
