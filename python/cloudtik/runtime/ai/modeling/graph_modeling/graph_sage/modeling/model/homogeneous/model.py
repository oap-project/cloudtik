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
from dgl.nn import SAGEConv

import tqdm
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)


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


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, edge_subgraph, x):
        src, dst = edge_subgraph.edges()
        return self.decoder(x[src] * x[dst])


class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_size, num_layers):
        super().__init__()

        self.hidden_size = hidden_size

        # encoder is a 1-layer GraphSAGE model
        self.encoder = GraphSAGE(
            in_feats, hidden_size, hidden_size, num_layers, F.relu, "mean"
        )
        self.decoder = EdgeDecoder(hidden_size)

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
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
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
            y = torch.empty(
                g.num_nodes(),
                self.encoder.hidden_size if l != len(
                    self.encoder.layers) - 1 else self.encoder.out_feats,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            x = x.to(device) if x is not None else x
            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    if x is not None:
                        h = x[input_nodes]
                    else:
                        # The first layer will come to here if x is None
                        h = self.get_encoder_inputs(g, input_nodes, blocks)
                    h = h.to(device)
                    h = layer(blocks[0], h)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                x = y
        return y

    def inference_nodes(self, g, x, nids, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set (feature or None if using node id)
        nids : Tensor of the indices of the node ids to be computed
        """
        # for batch, we don't compute representations layer by layer
        # instead we do it with batch and then layers
        num_nodes = nids.size(dim=0)
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

        y = torch.empty(
            num_nodes,
            self.encoder.out_feats,
            device=buffer_device,
            pin_memory=pin_memory,
        )
        y_start = 0

        x = x.to(device) if x is not None else x
        # first iterating over the mini batches
        with dataloader.enable_cpu_affinity():
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
            ):
                if x is not None:
                    h = x[input_nodes]
                else:
                    h = self.get_encoder_inputs(g, input_nodes, blocks)
                h = h.to(device)
                for l, layer in enumerate(self.encoder.layers):
                    h = layer(blocks[l], h)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)

                # store the final output
                num_output = output_nodes.size(dim=0)
                y[y_start:y_start+num_output] = h.to(buffer_device)
                y_start += num_output

        return y
