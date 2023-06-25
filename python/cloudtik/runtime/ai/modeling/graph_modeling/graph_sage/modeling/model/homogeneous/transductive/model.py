# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.model import GraphSAGEModel


class TransductiveGraphSAGEModel(GraphSAGEModel):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__(hidden_size, hidden_size, num_layers)

        # node embedding
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        return super().forward(
            pair_graph, neg_pair_graph, blocks, h)

    def get_input_embeddings(self):
        return self.emb.weight.data

    def get_inputs(self, input_nodes, blocks):
        return input_nodes

    def get_inference_inputs(self, g):
        return self.get_input_embeddings()

    def inference(self, g, x, device, batch_size):
        """Layer-wise inference algorithm to compute node embeddings.
        for the nodes of the entire graph.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.
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
            x = x.to(device)
            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    h = x[input_nodes]
                    h = layer(blocks[0], h)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                x = y
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
