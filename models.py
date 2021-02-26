import torch.nn as nn
from layers import GRUSubLayer, GGATLayer


class GGAT(nn.Module):
    def __init__(self, n_features, n_hid, n_class, dropout, alpha, n_heads):
        super(GGAT, self).__init__()

        """Create a public GRU sub layer for the three GGAT layers"""
        self.gru_public = GRUSubLayer(n_class)

        """Stack 3 GGAT layers"""
        self.net_a = GGATLayer(n_features, n_class, self.gru_public,
                               dropout, alpha, n_heads, require_gru=False)
        self.net_b = GGATLayer(n_class, n_class, self.gru_public,
                               dropout, alpha)
        self.net_c = GGATLayer(n_class, n_class, self.gru_public,
                               dropout, alpha)

    def forward(self, x, adj):
        """
        :param x: input features for the layer
        :param adj: adjacency matrix
        :return: result features of the layer
        """
        """GGAT 1"""
        x = self.net_a(x, adj)
        """GGAT 2"""
        x = self.net_b(x, adj)
        """GGAT 3"""
        x = self.net_c(x, adj)
        return x
