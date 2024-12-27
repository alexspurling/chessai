import gzip
import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenisation.charactertokeniser import encode
from tokenisation.decoder import decode
import time


# Hyperparameters
class Small:
    batch_size = 64   # How many independent sequences will we process in parallel?
    block_size = 128  # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = "cpu" if torch.cuda.is_available() else "cpu"
    eval_iters = 200
    n_embd = 256
    n_head = 4
    n_layer = 3
    dropout = 0.2


class Medium:
    batch_size = 64   # How many independent sequences will we process in parallel?
    block_size = 256  # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 100
    learning_rate = 3e-4
    eval_iters = 200
    n_embd = 512
    n_head = 8
    n_layer = 8
    dropout = 0.2


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, params):
        super().__init__()
        self.key = nn.Linear(params.n_embd, head_size, bias=False)
        self.query = nn.Linear(params.n_embd, head_size, bias=False)
        self.value = nn.Linear(params.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(params.block_size, params.block_size)))

        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities" / "weights")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, params):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, params) for _ in range(num_heads)])
        self.proj = nn.Linear(params.n_embd, params.n_embd)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(params.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, params):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, params)
        self.ffwd = FeedForward(n_embd, params)
        # Layer norm normalises values to mean = 0 and stddev = 1
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, params, device="cpu"):
        super().__init__()
        self.params = params
        self.device = device
        # each token directly reads off the logics for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, params.n_embd)
        self.position_embedding_table = nn.Embedding(params.block_size, params.n_embd)
        # i.e. 4 heads of 8-dimensional self-attention = 32 (same as n_embd)
        self.blocks = nn.Sequential(*[Block(params.n_embd, n_head=params.n_head, params=params) for _ in range(params.n_layer)])
        self.ln_f = nn.LayerNorm(params.n_embd)
        self.lm_head = nn.Linear(params.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = tok_emb + pos_emb  # (batch, time, channel)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
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
            idx_cond = idx[:, -self.params.block_size:]
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


class ChessModel:

    def __init__(self, model_state_file, vocab_size, params, device="cpu"):

        self.model_state_file = model_state_file
        self.params = params
        self.device = device

        print("Device is", device)

        self.model = BigramLanguageModel(vocab_size, device=device, params=params)

        if os.path.exists(model_state_file):
            print(f"Loading existing model from {model_state_file}")
            self.model.load_state_dict(torch.load(model_state_file, weights_only=True, map_location=device))
        self.m = self.model.to(device)

        # print the number of parameters in the model
        num_params = sum(p.numel() for p in self.m.parameters())
        print(f"{num_params / 1e6:.2f}M parameters")

    def train(self, tokendata):

        print("Opening training data")

        input_data = []
        line_count = 0

        with gzip.open(tokendata, "rt") as f:
            for line in f:
                input_data.extend(encode(line))
                line_count += 1
                if line_count % 100000 == 0:
                    print("Encoded lines: ", line_count, ". Total data: ", len(input_data) / 1e6, "MB")
                    break
                # if line_count == 5000000:
                #     break

        print("Training data loaded", len(input_data) / 1e6, "MB")

        print("Encoding...")

        data = torch.tensor(input_data, dtype=torch.long)

        print("Data encoded. Example data:")
        print(data.shape, data.dtype)
        print(data[:100])

        n = int(0.8 * len(data))
        train_data = data[:n]       # Train on the first 80%
        validation_data = data[n:]  # Validate on the last 20%

        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            # the target is the token that comes after the tokens in the batch
            data = train_data if split == "train" else validation_data
            ix = torch.randint(len(data) - self.params.block_size, (self.params.batch_size,))
            x = torch.stack([data[i:i+self.params.block_size] for i in ix])
            y = torch.stack([data[i+1:i+self.params.block_size+1] for i in ix])
            x, y = x.to(self.device), y.to(self.device)
            return x, y

        # Tell pytorch that we're not going to call backward() (backward-propagation) so it can be more memory efficient
        @torch.no_grad()
        def estimate_loss():
            out = {}
            self.model.eval()  # Set model to evaluation phase
            for split in ["train", "val"]:
                losses = torch.zeros(self.params.eval_iters)
                for k in range(self.params.eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train()  # Set model back to training phase
            return out

        # Adam is an optimizer like Stochastic Gradient Descent but apparently better
        # Learning rate 1e-3 can be changed?
        optimizer = torch.optim.AdamW(self.m.parameters(), lr=self.params.learning_rate)
        start_time = time.time()

        # Initialise training variables
        xb, yb = get_batch("train")
        logits, loss = self.m(xb, yb)

        for i in range(self.params.max_iters):

            # Every once in a while evaluate the loss on train and val sets
            if i % self.params.eval_interval == 0:
                losses = estimate_loss()
                print(f"step {i}: train loss {losses['train']:.4f}, val, loss {losses['val']:.4f}")
                # save current state of the model to disk
                torch.save(self.m.state_dict(), self.model_state_file)

            # sample a batch of data
            xb, yb = get_batch("train")

            # evaluate the loss
            logits, loss = self.m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("Final Loss:", loss.item())
        print("Time taken", (time.time() - start_time))

        for i in range(10):
            print(decode(self.generate()))

    def generate(self, start_with=None, num_moves_to_generate=2):
        start_with = start_with if start_with is not None else [33]
        idx = torch.tensor([start_with], dtype=torch.long, device=self.device)
        # Retrieve the first batch and convert it from a tensor into a python list
        all_batches = self.m.generate(idx, max_new_tokens=num_moves_to_generate)
        return all_batches[0].tolist()
