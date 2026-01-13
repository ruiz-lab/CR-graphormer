import torch
import torch.nn as nn
import torch.nn.functional as F

class SANLayer(nn.Module):
    """One SAN layer: Multi-head self-attention + FFN with edge mask"""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention with graph mask
        h, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(h))

        # Feed-forward network
        h = self.ffn(x)
        x = self.norm2(x + self.dropout(h))
        return x


class SAN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512,
                 num_layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SANLayer(hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        # Embed node features
        x = self.embedding(x)              # [N, F]
        x = x.unsqueeze(0)                 # [1, N, F] for MultiheadAttention

        # Build attention mask from edge_index if provided
        attn_mask = None
        if edge_index is not None:
            N = x.size(1)
            # Initialize all as -inf (blocked)
            attn_mask = torch.full((N, N), float('-inf'), device=x.device)
            # Allow edges in edge_index
            attn_mask[edge_index[0], edge_index[1]] = 0
            # Allow self-loops
            attn_mask[torch.arange(N), torch.arange(N)] = 0

        # Pass through SAN layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Back to [N, F]
        x = x.squeeze(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)  # compatible with F.nll_loss
