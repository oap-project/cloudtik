# Modifications Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

from contextlib import contextmanager

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv

import tqdm


class DistributedGraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type
    ):
        super(DistributedGraphSAGE, self).__init__()
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
        )  # activation None

    def forward(self, graphs, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, graphs)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h


class DistributedGraphSAGEModel(nn.Module):
    def __init__(self, vocab_size, hid_size, num_layers):
        super().__init__()

        self.hid_size = hid_size
        # node embedding
        self.emb = th.nn.Embedding(vocab_size, hid_size)

        # encoder is a 1-layer GraphSAGE model
        self.encoder = DistributedGraphSAGE(
            hid_size, hid_size, hid_size, num_layers, F.relu, "mean"
        )
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        h = self.encoder(blocks, h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.hid_size),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.encoder.layers):
            if i == len(self.encoder.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.hid_size),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.encoder.layers) - 1:
                    h = self.encoder.activation(h)
                #                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
