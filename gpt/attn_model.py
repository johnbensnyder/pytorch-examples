import torch

class Head(torch.nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size=128, n_embd=128, block_size=128, dropout=0.1, decoder=True):
        super().__init__()
        self.__dict__.update(locals())
        self.key = torch.nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = torch.nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = torch.nn.Linear(self.n_embd, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.decoder:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = torch.nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(torch.nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads=16, head_size=8, n_embd=128, dropout=0.1):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(torch.nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(torch.nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout=0.1):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleGPT(torch.nn.Module):

    def __init__(self, vocab_size, n_head=16, n_embd=128, n_layer=16, block_size=128, dropout=0.1, decoder=True):
        super().__init__()
        self.__dict__.update(locals())
        assert n_embd%n_head==0
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = torch.nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = torch.nn.Embedding(self.block_size, self.n_embd)
        #self.sa_heads = MultiHeadAttention(num_heads=4, head_size=self.n_embd//4, n_embd=self.n_embd)
        #self.ffwd = FeedFoward(self.n_embd, self.dropout)
        self.blocks = torch.nn.Sequential(*[Block(self.n_embd, n_head=self.n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(self.n_embd) # final layer norm
        self.lm_head = torch.nn.Linear(self.n_embd, self.vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device='cuda'))
        x = tok_emb + pos_emb
        #x = self.sa_heads(x)
        #x = self.ffwd(x)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to block size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
