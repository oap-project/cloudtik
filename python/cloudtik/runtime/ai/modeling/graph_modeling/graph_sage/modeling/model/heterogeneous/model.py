"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Chen Haifeng
"""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, HeteroGraphConv


class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type,
            relations,
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.relations = relations

        # Heterogeneous
        # input layer
        self.layers.append(HeteroGraphConv({
            relation: SAGEConv(in_feats, hidden_size, aggregator_type)
            for relation in relations}))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(HeteroGraphConv({
                relation: SAGEConv(hidden_size, hidden_size, aggregator_type)
                for relation in relations}))
        # output layer
        self.layers.append(
            HeteroGraphConv({
                relation: SAGEConv(hidden_size, out_feats, aggregator_type)
                for relation in relations})
        )
        self.hidden_size = hidden_size
        self.out_feats = out_feats

    def forward(self, graphs, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, graphs)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                # Heterogeneous
                h = {k: self.activation(v) for k, v in h.items()}
        return h


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_size, relations):
        super().__init__()
        self.relations = relations

        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def apply_decoder(self, edges):
        h = self.decoder(edges.src['x'] * edges.dst['x'])
        return {'h': h}

    def forward(self, edge_subgraph, x):
        # Heterogeneous
        # x contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in HeteroGraphConv
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x  # assigns 'x' of all node types in one shot
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(self.apply_decoder, etype=etype)
            return edge_subgraph.edata['h']


class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_size, n_layers,
                 relations):
        super().__init__()

        self.hidden_size = hidden_size
        self.relations = relations

        # encoder is a 1-layer GraphSAGE model with n-layer of SAGECov
        self.encoder = GraphSAGE(in_feats, hidden_size, hidden_size, n_layers, F.relu, "mean",
                                 relations)

        # Heterogeneous edge decoder
        self.decoder = EdgeDecoder(
            self.hidden_size, self.relations)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.encoder(blocks, x)
        h_pos = self.decoder(pair_graph, h)
        h_neg = self.decoder(neg_pair_graph, h)
        return h_pos, h_neg

    def get_inputs(self, input_nodes, blocks):
        raise NotImplementedError("The final model must implement this method.")

    def get_inference_inputs(self, g):
        raise NotImplementedError("The final model must implement this method.")
