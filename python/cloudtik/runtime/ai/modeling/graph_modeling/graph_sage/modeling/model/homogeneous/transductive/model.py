# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)
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
    def __init__(self, vocab_size, hidden_size, n_layers):
        super().__init__()

        self.hidden_size = hidden_size

        # node embedding
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)
        # encoder is a 1-layer GraphSAGE model
        self.encoder = GraphSAGE(hidden_size, hidden_size, hidden_size, n_layers, F.relu, "mean")
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # cosine similarity with linear
        # self.predictor=dglnn.EdgePredictor('cos',hidden_size,1)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        h = self.encoder(blocks, h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # h_pos = self.predictor(h[pos_src], h[pos_dst])
        # h_neg = self.predictor(h[neg_src], h[neg_dst])
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute node embeddings.
        for the nodes of the entire graph.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.

        # feat = g.ndata['feat']
        # use pretrained embedding as node features
        feat = self.emb.weight.data
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
            feat = feat.to(device)
            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                feat = y
        return y

    def inference_nodes(self, g, nids, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        nids : Tensor of the indices of the node ids to be computed
        """
        # for batch, we don't compute representations layer by layer
        # instead we do it with batch and then layers

        # feat = g.ndata['feat']
        # use pretrained embedding as node features
        num_nodes = nids.size(dim=0)
        if batch_size == 0:
            batch_size = num_nodes

        feat = self.emb.weight.data
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
        # first iterating over the mini batches
        with dataloader.enable_cpu_affinity():
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
            ):
                h = feat[input_nodes]
                for l, layer in enumerate(self.encoder.layers):
                    h = layer(blocks[l], h)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)

                # store the final output
                num_output = output_nodes.size(dim=0)
                y[y_start:y_start+num_output] = h.to(buffer_device)
                y_start += num_output

        return y
