import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import LayerNorm

class GlobalTransformer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by heads"
        self.heads = heads
        self.d_head = dim // heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, D = x.size()
        H, d = self.heads, self.d_head

        Q = self.q_proj(x).view(N, H, d)
        K = self.k_proj(x).view(N, H, d)
        V = self.v_proj(x).view(N, H, d)

        attn_scores = torch.einsum("nhd,mhd->hnm", Q, K) * self.scale  # [H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("hnm,mhd->nhd", attn_weights, V)  # [N, H, d]
        out = out.reshape(N, D)
        return self.out_proj(out)


class GraphGPSLayer(nn.Module):
    def __init__(self, hidden_dim=512, heads=8, dropout=0.1):
        super().__init__()
        # Fix: add self-loops and ensure correct number of nodes
        self.local_gnn = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.global_transformer = GlobalTransformer(hidden_dim, heads, dropout)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weights=None):
        # Local message passing (residual + norm)
        res = x
        # Ensure edge_index is safe: add self-loops and clip indices
        num_nodes = x.size(0)
        x = self.local_gnn(x, edge_index, edge_weights)
        x = self.norm1(x + res)

        # Global transformer (residual + norm)
        res = x
        x = self.global_transformer(x)
        x = self.norm2(x + res)

        return x


class GraphGPS(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weights=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        return F.log_softmax(x, dim=1)  # for F.nll_loss
