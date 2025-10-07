import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, heads=8, dropout=0.1):
        super(GAT, self).__init__()
        # First GAT layer (multi-head attention)
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # Second GAT layer (averaging heads)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # Second layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        # Log softmax -> compatible with F.nll_loss
        return F.log_softmax(x, dim=1)
