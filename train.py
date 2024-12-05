import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenisation.decoder import decode
import time

# Hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel?
block_size = 8   # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
vocab_size = 256

print("Device", device)

def train():

    with open("../../rust/chessai/tokens.bin", "rb") as f:
        raw_data = f.read(1000000)

    data = torch.tensor(list(raw_data), dtype=torch.long)
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
    print("inputs:")
    print(xb.shape)
    print(xb)
    print("targets:")
    print(yb.shape)
    print(yb)

    print("----")

    for b in range(batch_size):  # batch dimension
        for t in range(block_size):  # time dimension
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"when input is {context.tolist()} the target is: {target}")

    model = BigramLanguageModel(vocab_size)
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logics for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (batch, time, channel)

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
            # get the predictions
            logits, loss = self(idx)
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
