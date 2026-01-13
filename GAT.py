class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, heads=8, dropout=0.1, concat=True):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout))
        hidden_out_dim = hidden_dim * heads if concat else hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_out_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout))
        self.convs.append(GATConv(hidden_out_dim, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)