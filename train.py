import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenisation.decoder import decode
import time

# Hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

vocab_size = 256


def train():

    with open("../../rust/chessai/tokens.bin", "rb") as f:
        raw_data = f.read(1000000)

    data = torch.tensor(list(raw_data), dtype=torch.long)

    print("Example data")
    print(data.shape, data.dtype)
    print(data[:100])

    n = int(0.9 * len(data))
    train_data = data[:n]       # Train on the first 90%
    validation_data = data[n:]  # Validate on the last 10%

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        # the target is the token that comes after the tokens in the batch
        data = train_data if split == "train" else validation_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    xb, yb = get_batch("train")
    # print("inputs:")
    # print(xb.shape)
    # print(xb)
    # print("targets:")
    # print(yb.shape)
    # print(yb)
    #
    # print("----")

    model = BigramLanguageModel()
    m = model.to(device)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    # Tell pytorch that we're not going to call backward() (backward-propagation) so it can be more memory efficient
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()  # Set model to evaluation phase
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()  # Set model back to training phase
        return out

    expected_loss = -math.log(1 / vocab_size)

    print("Device is", device)
    # Create 1x1 tensor holding zeros to hold indices?
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Retrieve the first batch and convert it from a tensor into a python list
    first_batch = m.generate(context, max_new_tokens=100)[0].tolist()
    output = decode(first_batch)

    # Outputs garbage because we haven't trained the model yet!
    print("Untrained random output: ")
    print(output)

    # Adam is an optimizer like Stochastic Gradient Descent but apparently better
    # Learning rate 1e-3 can be changed?
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    start_time = time.time()
    for iter in range(max_iters):

        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val, loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train") # Not sure why we change the batch size to 32 here

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Final Loss:", loss.item(), "Expected loss:", expected_loss)
    print("Time taken", (time.time() - start_time))

    print("Device is", device)
    # Create 1x1 tensor holding zeros to hold indices?
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Retrieve the first batch and convert it from a tensor into a python list
    first_batch = m.generate(idx, max_new_tokens=100)[0].tolist()
    output = decode(first_batch)

    # Outputs garbage because we haven't trained the model yet!
    print(output)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities" / "weights")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logics for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # i.e. 4 heads of 8-dimensional self-attention = 32 (same as n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (batch, time, channel)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (batch, time, vocab_size)

        if targets is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            # Flatten the arrays from 3 dimensions to 2 so it's in a form that the cross_entropy function expects
            logits = logits.view(batch * time, channel)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


train()
