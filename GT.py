import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GTLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.1, edge_dim=None):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.lin_q = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_k = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_v = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.lin_out = nn.Linear(out_dim * heads, out_dim)

        self.edge_emb = nn.Linear(edge_dim, heads) if edge_dim is not None else None

    def forward(self, x, edge_index, edge_attr=None):
        Q = self.lin_q(x).view(-1, self.heads, self.out_dim)
        K = self.lin_k(x).view(-1, self.heads, self.out_dim)
        V = self.lin_v(x).view(-1, self.heads, self.out_dim)

        return self.propagate(edge_index, Q=Q, K=K, V=V, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

    def message(self, Q_i, K_j, V_j, index, ptr, size_i, edge_attr):
        alpha = (Q_i * K_j).sum(dim=-1) / (self.out_dim ** 0.5)
        if self.edge_emb is not None and edge_attr is not None:
            alpha = alpha + self.edge_emb(edge_attr)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)
        return V_j * alpha .unsqueeze(-1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_dim)
        return self.lin_out(aggr_out)


class GT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, heads=8, dropout=0.1, edge_dim=None):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GTLayer(hidden_dim, hidden_dim, heads, dropout, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None, pe=None):
        if pe is not None:
            x = x + pe
        x = self.embedding(x)
        for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            x_res = x
            x = layer(x, edge_index, edge_attr)
            x = norm(x + x_res)
            x = x + ffn(x)
        return F.log_softmax(self.mlp_out(x))