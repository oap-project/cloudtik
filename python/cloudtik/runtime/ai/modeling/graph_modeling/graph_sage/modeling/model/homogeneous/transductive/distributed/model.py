# Modifications Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

from contextlib import contextmanager

import numpy as np
import torch as th
import dgl

import tqdm

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.transductive.model import TransductiveGraphSAGEModel


class DistTransductiveGraphSAGEModel(TransductiveGraphSAGEModel):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__(vocab_size, hidden_size, num_layers)

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
            (g.num_nodes(), self.encoder.hidden_size),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.encoder.layers):
            if i == len(self.encoder.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.encoder.out_feats),
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
                    # h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
