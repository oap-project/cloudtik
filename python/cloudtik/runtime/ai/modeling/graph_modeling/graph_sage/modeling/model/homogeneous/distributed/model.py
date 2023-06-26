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

from contextlib import contextmanager

import numpy as np
import torch as th
import dgl

import tqdm

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.model import GraphSAGEModel


class DistGraphSAGEModel(GraphSAGEModel):
    def __init__(self, in_feats, hidden_size, num_layers):
        super().__init__(in_feats, hidden_size, num_layers)

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
                if x is not None:
                    h = x[input_nodes]
                else:
                    # The first layer will come to here if x is None
                    h = self.get_encoder_inputs(input_nodes, blocks)
                h = h.to(device)
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
