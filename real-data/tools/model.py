import torch
import torch.nn as nn
from torch.nn import functional
import math


class SelfAttention(nn.Module):
    """
    The multihead attention model
    """

    def __init__(self, config):
        super().__init__()
        self.attn_matrix = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.linear = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask, device):
        # The batch size, sequence length, and embedding dim
        B, T, C = x.size()

        # Calculate query, key, and values for all heads
        qkv = self.attn_matrix(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Get the attention results
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_bias = torch.zeros(B, T, T, dtype=q.dtype).to(device)
        mask = mask.view(B, 1, T) & mask.view(B, T, 1)
        mask[:, :, 0] = True
        attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        attn_weight += attn_bias.view(B, 1, T, T)
        attn_weight = torch.softmax(attn_weight, dim=-1).nan_to_num(
            nan=0, posinf=0, neginf=0
        )
        attn_weight = torch.dropout(
            attn_weight, self.dropout if self.training else 0, train=self.training
        )
        y = attn_weight @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Linear layer
        y = self.resid_dropout(self.linear(y))
        return y


class FeedForward(nn.Module):
    """
    The fully connected feed-forward network
    """

    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.n_embd, config.n_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(config.n_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization layer
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return functional.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        )


class Block(nn.Module):
    """
    A single block for the transformer model
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x, mask, device):
        x = self.ln1(x + self.attn(x, mask, device))
        x = self.ln2(x + self.mlp(x))
        return x


class TransformerClassifier(nn.Module):
    """
    The full transforemr classification model
    """

    def __init__(self, config):
        super().__init__()
        self.emb_static = nn.Embedding(config.vocab_size, config.n_embd)
        self.emb_pos = nn.Embedding(config.max_length, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_blocks)])
        self.agg = config.final_rep_agg
        self.lin = nn.Linear(config.n_embd, 1, bias=config.bias)

    def forward(self, x, device):
        # Embed
        B, T = x.size()
        mask = (x != 0).to(device)
        x = self.emb_static(x) + self.emb_pos(torch.arange(0, T).to(device))

        # Pass through the blocks
        for block in self.blocks:
            x = block(x, mask, device)

        # Take the first token
        x = x[:, 0, :]

        # Get the final probability
        x = self.lin(x)
        x = functional.sigmoid(x)
        return torch.flatten(x, start_dim=-2, end_dim=-1)
