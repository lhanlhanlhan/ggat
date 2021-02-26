import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GGATLayer(nn.Module):
    def __init__(self, in_features, out_features, gru_layer,
                 gat_dropout, gat_alpha, gat_heads=1,
                 require_gat=True, require_gru=True):
        """
        A layer with GRU and GAT
        :param in_features: dimension of the input feature vector
        :param out_features: dimension of the output feature vector
        :param gat_dropout: dropout
        :param gat_alpha: alpha value of leaky_relu
        :param gat_heads: number of heads in GAT
        """
        super(GGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gat_dropout = gat_dropout
        self.gat_alpha = gat_alpha
        self.gat_heads = gat_heads
        self.require_gat = require_gat
        self.require_gru = require_gru

        if (require_gat is False) and (require_gru is False):
            raise Exception("At least one sublayer shall be defined.")

        """GAT"""
        if require_gat:
            if gat_heads > 1:
                self.attentions = [GATSubLayer(in_features, out_features,
                                               dropout=gat_dropout, alpha=gat_alpha, concat=True)
                                   for _ in range(gat_heads)]
                for i, attention in enumerate(self.attentions):
                    self.add_module('gat_attention_{}'.format(i), attention)
                """There has to be one output GAT transform layer if the head number is larger than 1."""
                self.gat_out_att = GATSubLayer(out_features * gat_heads, out_features,
                                               dropout=gat_dropout, alpha=gat_alpha, concat=False)
                self.add_module('gat_out_attention', self.gat_out_att)
            else:
                self.gat_out_att = GATSubLayer(in_features, out_features,
                                               dropout=gat_dropout, alpha=gat_alpha, concat=False)
                self.add_module('gat_out_attention', self.gat_out_att)

        """Transform: to change dimension of the input vector if needed by GRU"""
        if in_features != out_features and require_gru:
            self.transform = nn.Linear(in_features, out_features)

        """GRU: to store the global GRU"""
        if require_gru:
            self.gru_layer = gru_layer

    def forward(self, h_in, adj):
        """
        :param h_in: input hidden layer representation
        :param adj: adjacency matrix
        :return: output hidden layer representation
        """

        """Transform: to change dimension of the input vector if needed by GRU"""
        gru_in = self.transform(h_in) if (self.in_features != self.out_features and self.require_gru) else h_in
        x = F.dropout(h_in, self.gat_dropout, training=self.training)

        """GAT"""
        if self.require_gat:
            if self.gat_heads > 1:
                outputs = []
                for att in self.attentions:
                    # attention for the current head
                    o = att(x, adj)
                    outputs.append(o)
                # concatenation of the current-head attention
                x = torch.cat(outputs, dim=1)

                x = F.dropout(x, self.gat_dropout, training=self.training)

            x = F.elu(self.gat_out_att(x, adj))

            x = F.log_softmax(x, dim=1)

        """GRU filter"""
        if self.require_gru:
            x = F.dropout(x, self.gat_dropout, training=self.training) if self.require_gat else x
            x = self.gru_layer(x, gru_in)
            x = F.log_softmax(x, dim=1)
        return x


class GRUSubLayer(nn.Module):
    def __init__(self, n_features):
        """
        GRU sub layer
        :param n_features: number of features
        """
        super(GRUSubLayer, self).__init__()
        self.n_features = n_features

        """GRU: Reset Gate"""
        self.reset_gate = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            # 激活函数为 sigmoid
            nn.Sigmoid()
        )
        """GRU: Update Gate"""
        self.update_gate = nn.Sequential(
            # 线性变换就相当于矩阵乘
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid()
        )
        """GRU: The output transform"""
        self.transform = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Tanh()
        )

    def forward(self, h, h_in):
        a = torch.cat((h, h_in), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat((h, r * h_in), 1)
        h_hat = self.transform(joined_input)

        output = (1 - z) * h_in + z * h_hat
        return output


class GATSubLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATSubLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        """GAT W & a"""

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        """GAT: LeakyRelu"""
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
