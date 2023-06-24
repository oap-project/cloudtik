# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(
            SAGEConv(hidden_size, out_feats, aggregator_type)
        )
        self.hidden_size = hidden_size
        self.out_feats = out_feats

    def forward(self, graphs, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, graphs)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h


class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size

        # encoder is a 1-layer GraphSAGE model
        self.encoder = GraphSAGE(
            in_feats, hidden_size, hidden_size, num_layers, F.relu, "mean"
        )
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.encoder(blocks, x)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg
