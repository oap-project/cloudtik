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

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, HeteroGraphConv

import tqdm
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import get_node_indices, get_num_nodes_of_graph, get_total_num_nodes, \
    get_num_nodes_of_nids, tensor_dict_new, tensor_dict_to, tensor_dict_store, \
    tensor_dict_collect, tensor_dict_store_at


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

    def get_inputs(self, g, input_nodes, blocks):
        raise NotImplementedError("The final model must implement this method.")

    def get_inference_inputs(self, g):
        raise NotImplementedError("The final model must implement this method.")

    def get_encoder_inputs(self, g, input_nodes, blocks):
        return self.get_inputs(g, input_nodes, blocks)

    def inference(self, g, x, device, batch_size):
        """Layer-wise inference algorithm to compute node embeddings.
        for the nodes of the entire graph.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set (feature or None if using node id)
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.

        sampler = MultiLayerFullNeighborSampler(1)

        # Heterogeneous: create node indices based on the relations
        node_indices = get_node_indices(
            g, self.relations)

        dataloader = DataLoader(
            g,
            node_indices,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        # compute representations layer by layer
        for l, layer in enumerate(self.encoder.layers):
            out_size = self.encoder.hidden_size if l != len(
                        self.encoder.layers) - 1 else self.encoder.out_feats
            num_nodes = get_num_nodes_of_graph(g, self.relations)
            y = tensor_dict_new(
                num_nodes, out_size,
                buffer_device, pin_memory)

            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    # Heterogeneous: everything is a dict here
                    # input_nodes is a dict, output_nodes is dict
                    # input_block.srcdata[dgl.NID] is a dict
                    if x is not None:
                        h = tensor_dict_collect(x, input_nodes)
                    else:
                        # The first layer will come to here if x is None
                        h = self.get_encoder_inputs(g, input_nodes, blocks)
                    h = tensor_dict_to(h, device)
                    h = layer(blocks[0], h)
                    if l != len(self.encoder.layers) - 1:
                        h = {k: F.relu(v) for k, v in h.items()}

                    tensor_dict_store(
                        h, y, output_nodes, buffer_device)
                x = y
        return y

    def inference_nodes(self, g, x, nids, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set (feature or None if using node id)
        nids : Tensor or dict[ntype, Tensor] of the indices of the node ids to be computed
        """
        # for batch, we don't compute representations layer by layer
        # instead we do it with batch and then layers
        num_nodes = get_total_num_nodes(nids)
        if batch_size == 0:
            batch_size = num_nodes

        sampler = MultiLayerFullNeighborSampler(
            num_layers=self.encoder.layers)
        dataloader = DataLoader(
            g,
            nids.to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        num_nodes = get_num_nodes_of_nids(g, self.relations, nids)
        y = tensor_dict_new(
                num_nodes, self.encoder.out_feats,
                buffer_device, pin_memory)
        y_start = {node_type: 0 for node_type in y}

        # first iterating over the mini batches
        with dataloader.enable_cpu_affinity():
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
            ):
                # Inductive: use node feature or id
                # Heterogeneous: srcdata will return a dict
                if x is not None:
                    h = tensor_dict_collect(x, input_nodes)
                else:
                    h = self.get_encoder_inputs(g, input_nodes, blocks)
                h = tensor_dict_to(h, device)
                for l, layer in enumerate(self.encoder.layers):
                    h = layer(blocks[l], h)
                    if l != len(self.encoder.layers) - 1:
                        h = {k: F.relu(v) for k, v in h.items()}

                # store the final output
                tensor_dict_store_at(
                    h, y, output_nodes, y_start, buffer_device)

        return y
