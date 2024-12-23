import torch
import torch.nn as nn
import torch.optim as optim

# 1) Toy data
text = "hello world this is a toy example hello world again"
vocab = list(set(text.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
encoded = [word2idx[w] for w in text.split()]

# 2) Create sequences
SEQ_LEN = 3
data = []
targets = []
for i in range(len(encoded) - SEQ_LEN):
    data.append(encoded[i:i+SEQ_LEN])
    targets.append(encoded[i+SEQ_LEN])

data = torch.tensor(data, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

# 3) Small model (e.g., LSTM)
class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x shape: [batch, seq_len]
        x = self.embed(x)  # [batch, seq_len, embed_dim]
        output, (h, c) = self.lstm(x)
        # take the last hidden state
        last_output = output[:, -1, :]  # [batch, hidden_dim]
        logits = self.fc(last_output)   # [batch, vocab_size]
        return logits

model = TinyLSTM(vocab_size=len(vocab), embed_dim=8, hidden_dim=16)

# 4) Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 100
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    logits = model(data)            # [batch, vocab_size]
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5) Generate text
def generate_text(prompt, length=10):
    words = prompt.split()
    for _ in range(length):
        x = torch.tensor([[word2idx[w] for w in words[-SEQ_LEN:]]], dtype=torch.long)
        logits = model(x)
        probs = nn.Softmax(dim=-1)(logits[0])
        # sample from probabilities
        next_idx = torch.multinomial(probs, 1).item()
        words.append(idx2word[next_idx])
    return " ".join(words)

print(generate_text("this is", length=5))