import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch_geometric.utils import to_dense_adj

class AnchorPositionalEncoding(nn.Module):
    """
    Computes anchor-based positional encoding (APE) for each node.
    """
    def __init__(self, num_anchors, hidden_dim):
        super().__init__()
        self.num_anchors = num_anchors
        self.anchor_emb = nn.Parameter(torch.randn(num_anchors, hidden_dim))

    def forward(self, adj):
        """
        adj: [N, N] dense adjacency matrix
        Returns: [N, hidden_dim] anchor-based positional encoding
        """
        # Compute anchor distances (here simplified as degree similarity)
        deg = adj.sum(dim=-1, keepdim=True)
        sim = torch.mm(deg, deg.t()) / (torch.norm(deg) + 1e-6)
        anchor_weights = torch.softmax(sim[:, :self.num_anchors], dim=-1)
        return anchor_weights @ self.anchor_emb  # [N, hidden_dim]

class Gophormer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, heads=8,
                 num_anchors=4, num_global_tokens=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_global_tokens = num_global_tokens

        # Node feature embedding
        self.node_embedding = nn.Linear(in_dim, hidden_dim)

        # Anchor-based positional encoding
        self.ape = AnchorPositionalEncoding(num_anchors=num_anchors, hidden_dim=hidden_dim)

        # Learnable global tokens
        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, hidden_dim))

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        """
        N = x.size(0)
        device = x.device

        # Dense adjacency matrix (for anchor encoding and attention mask)
        adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0).to(device)
        adj = (adj + torch.eye(N, device=device)).clamp(max=1)

        # Node embedding + anchor positional encoding
        h = self.node_embedding(x) + self.ape(adj)  # [N, hidden_dim]

        # Construct full token matrix (global + node)
        global_tokens = self.global_tokens.expand(1, -1, -1)      # [1, T, hidden_dim]
        node_tokens = h.unsqueeze(0)                              # [1, N, hidden_dim]
        tokens = torch.cat([global_tokens, node_tokens], dim=1)   # [1, T+N, hidden_dim]

        # Attention mask (prevent attention between unconnected nodes)
        mask = (adj == 0).bool()  # [N, N]
        full_mask = torch.zeros((N + self.num_global_tokens, N + self.num_global_tokens), dtype=torch.bool, device=device)
        full_mask[self.num_global_tokens:, self.num_global_tokens:] = mask

        # Transformer
        tokens = self.transformer(tokens, src_key_padding_mask=None, mask=full_mask)

        # Extract node embeddings
        out = tokens[:, self.num_global_tokens:, :]  # [1, N, hidden_dim]
        out = self.out_proj(out.squeeze(0))          # [N, out_dim]
        return F.log_softmax(out)
