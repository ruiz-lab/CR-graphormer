import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, degree

class GraphormerLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # Multi-head QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # Norms and dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        """
        x: [N, dim]
        attn_bias: [heads, N, N]
        """
        N, D = x.size()
        # QKV projection
        qkv = self.qkv(x).reshape(N, 3, self.heads, D // self.heads)
        Q, K, V = qkv[:,0], qkv[:,1], qkv[:,2]  # [N, heads, head_dim]

        # Transpose to [heads, N, head_dim]
        Q, K, V = Q.permute(1,0,2), K.permute(1,0,2), V.permute(1,0,2)

        # Scaled dot-product attention + SPD/edge bias
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V).permute(1,0,2).reshape(N, D)
        x = self.norm1(x + self.dropout(self.out_proj(out)))

        # Feed-forward
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class Graphormer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2,
                 heads=8, max_dist=5, dropout=0.1, add_global_token=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.max_dist = max_dist
        self.add_global_token = add_global_token

        # Input / output projections
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Graphormer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, heads, dropout) for _ in range(num_layers)
        ])

        # Embeddings
        self.dist_emb = nn.Embedding(max_dist + 1, heads)
        self.degree_emb = nn.Embedding(512, hidden_dim)  # assume max degree 512

        if add_global_token:
            self.global_token = nn.Parameter(torch.randn(1, hidden_dim))

    def compute_spd(self, edge_index, num_nodes):
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        dist = torch.full((num_nodes, num_nodes), float('inf'), device=adj.device)
        dist[adj > 0] = 1
        dist[torch.eye(num_nodes, dtype=torch.bool, device=adj.device)] = 0

        for k in range(num_nodes):
            dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

        dist[torch.isinf(dist)] = self.max_dist
        return dist.clamp(0, self.max_dist).long()

    def degree_encoding(self, edge_index, num_nodes):
        deg = degree(edge_index[0], num_nodes=num_nodes).long()
        deg = torch.clamp(deg, 0, 511)
        return self.degree_emb(deg)

    def forward(self, x, edge_index):
        N = x.size(0)
        x = self.input_proj(x)

        # Add degree encoding
        x = x + self.degree_encoding(edge_index, N)

        # Add global token if needed
        if self.add_global_token:
            global_tok = self.global_token.expand(1, -1)  # [1, hidden_dim]
            x = torch.cat([global_tok, x], dim=0)
            offset = 1
        else:
            offset = 0

        # SPD bias
        spd = self.compute_spd(edge_index, N)
        attn_bias = self.dist_emb(spd).permute(2,0,1)  # [heads, N, N]

        # Add zeros for global token
        if self.add_global_token:
            attn_bias = F.pad(attn_bias, (offset,0,offset,0))  # [heads, N+1, N+1]

        # Forward through layers
        for layer in self.layers:
            x = layer(x, attn_bias)

        # Output
        if self.add_global_token:
            x = x[1:]  # remove global token for node outputs

        out = self.output_proj(x)
        return F.log_softmax(out, dim=-1)
