import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, softmax
import networkx as nx

def build_expander_edges(num_nodes, degree=3, device='cpu'):
    # Ensure n*d is even
    if (num_nodes * degree) % 2 != 0:
        degree = degree - 1 if degree > 1 else 2  # make n*d even
    G = nx.random_regular_graph(d=degree, n=num_nodes)
    edges = torch.tensor(list(G.edges), dtype=torch.long, device=device)
    return torch.cat([edges.t(), edges.flip(0).t()], dim=1)  # bidirectional

class SparseAttention(nn.Module):
    """Sparse attention over local + expander + global edges."""
    def __init__(self, dim, heads=8, dropout=0.1, num_global_tokens=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))

    def forward(self, x, edge_index, expander_edge_index):
        N = x.size(0)
        H, D = self.heads, self.head_dim

        # Add global tokens
        global_tokens = self.global_tokens  # shape [num_global_tokens, dim]
        x_full = torch.cat([x, global_tokens], dim=0)  # [N + G, dim]

        # Compute Q, K, V
        qkv = self.qkv(x_full).view(N + self.num_global_tokens, 3, H, D)
        Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [N+G, H, D]

        # Build sparse edge set: local + expander + global bidirectional
        local_edges = add_self_loops(edge_index, num_nodes=N)[0]
        expander_edges = expander_edge_index
        global_nodes = torch.arange(N, N + self.num_global_tokens, device=x.device)

        # Node → global and global → node edges
        node_to_global = torch.stack(torch.meshgrid(torch.arange(N, device=x.device), global_nodes, indexing='ij'), dim=0).reshape(2, -1)
        global_to_node = node_to_global.flip(0)

        combined_edges = torch.cat([local_edges, expander_edges, node_to_global, global_to_node], dim=1)
        src, dst = combined_edges

        # Compute attention scores per head
        q = Q[dst]  # [num_edges, H, D]
        k = K[src]
        v = V[src]
        scores = (q * k).sum(dim=-1) * self.scale  # [num_edges, H]

        # Per-node softmax
        attn = softmax(scores, dst)  # [num_edges, H]
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.zeros_like(Q)
        out.index_add_(0, dst, attn.unsqueeze(-1) * v)
        out = out[:N]  # discard global tokens

        # Concatenate heads
        out = out.reshape(N, H * D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class ExphormerLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, hidden_mult=4, num_global_tokens=4, expander_degree=3):
        super().__init__()
        self.attn = SparseAttention(dim, heads, dropout, num_global_tokens)
        self.ffn = FeedForward(dim, hidden_mult * dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.expander_degree = expander_degree

    def forward(self, x, edge_index):
        expander_edges = build_expander_edges(x.size(0), self.expander_degree, x.device)

        # Sparse attention + residual
        h = self.attn(self.norm1(x), edge_index, expander_edges)
        x = x + self.dropout(h)

        # Feedforward + residual
        h2 = self.ffn(self.norm2(x))
        x = x + self.dropout(h2)
        return x


class Exphormer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, heads=8,
                 dropout=0.1, num_global_tokens=4, expander_degree=3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ExphormerLayer(hidden_dim, heads, dropout, hidden_mult=4,
                           num_global_tokens=num_global_tokens, expander_degree=expander_degree)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.norm(x)
        return F.log_softmax(self.output_proj(x), dim=1)
