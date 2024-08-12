import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-.5)# (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # now we can perform the weighter aggregation
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttentionConcat(nn.Module):
    """ Implements multi-head self-attention using individual heads and concatenating their results at the end """
    def __init__(self, num_heads, head_size, n_embd, device, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], -1)
        out = self.dropout(self.proj(out))
        return out

class MultiHeadAttention(nn.Module):
    """implements multi-headed masked self-attention using tensor operations"""

    def __init__(self, num_heads, head_size, n_embd, device, block_size, dropout=0.2):
        super().__init__()
        gen = torch.Generator(device=device)
        gen.manual_seed(42)

        self.num_heads = num_heads
        self.n_embd = n_embd
        self.device = device
        self.block_size = block_size
        self.dropout = dropout
        self.head_size = head_size

        key = (
            torch.randn(num_heads, n_embd, head_size, generator=gen, device=device)
            * (n_embd * num_heads) ** -0.5
        )
        query = (
            torch.randn(num_heads, n_embd, head_size, generator=gen, device=device)
            * (n_embd * num_heads) ** -0.5
        )
        value = (
            torch.randn(num_heads, n_embd, head_size, generator=gen, device=device)
            * (n_embd * num_heads) ** -0.5
        )

        self.key = nn.Parameter(key)
        self.query = nn.Parameter(query)
        self.value = nn.Parameter(value)

        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

        # (B, T, n_embd) x (num_heads, n_embd, head_size) --> (B, num_heads, T, head_size)
        self.register_buffer(
            "tril", torch.tril(torch.ones(num_heads, block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        x = x.unsqueeze(1)  # (batch_size, 1, context_length, n_embd)
        k = x @ self.key  # (batch_size, num_heads, context_length, head_size)
        q = x @ self.query  # (batch_size, num_heads, context_length, head_size)

        wei = (
            q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        )  # (bs, nh, cl, hs) x (bs, nh, hs, cl) -> (bs, nh, cl, cl)
        wei = wei.masked_fill(self.tril[:, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (bs, nh, cl, cl)

        v = x @ self.value  # (bs, 1, cl, ne) x (nh, ne, hs) -> (bs, nh, cl, hs)
        out = wei @ v  # (bs, nh, cl, cl) x (bs, nh, cl, hs) -> (bs, nh, cl, hs)
        out = out.permute(0, 2, 1, 3)  # (bs, cl, nh, hs)
        out = out.reshape(
            out.shape[0], out.shape[1], self.n_embd
        )  # (bs, cl, n_embd) = (B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer with non-linearity included"""

    def __init__(self, n_embd, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd, n_embd
            ),  # the projection to go back to the resudial pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_head, n_embd, device, block_size, dropout=0.2):
        super().__init__()
        self.head_size = n_embd // n_head
        self.sa = MultiHeadAttentionConcat(n_head, self.head_size, n_embd, device, block_size, dropout=0.2)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_blocks, device, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.device = device
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # every token has a position embedding
        self.blocks = nn.Sequential(
            *[Block(n_head=n_head, n_embd=n_embd, device=device, block_size=block_size, dropout=dropout) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # the final layer norm
        self.ln_head = nn.Linear(n_embd, vocab_size)  # the language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.ln_head(x)  # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array, where T is the context length
        for _ in range(max_new_tokens):
            idx_cropped = idx[:, -self.block_size:]
            logits, loss = self(idx_cropped)  # out: (B, T, C)
            # pick the last context window to sample the next token
            logits = logits[:, -1, :]  # (B, C)
            # map the outputs to a probability dist
            logits = F.softmax(logits, dim=1)
            # sample the next index
            next_idx = torch.multinomial(logits, 1)  # (B, 1)
            # concatenate the current context with the sampled one
            idx = torch.concat((idx, next_idx), dim=1)  # (B, T+1)

        return idx

