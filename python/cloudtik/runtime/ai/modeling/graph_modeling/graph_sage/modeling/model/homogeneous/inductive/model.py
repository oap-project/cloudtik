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
import torch.nn.functional as F
import tqdm
import dgl
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
)

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.model import GraphSAGEModel


class InductiveGraphSAGEModel(GraphSAGEModel):
    def __init__(self, in_feats, hidden_size, num_layers, node_feature):
        # Inductive: embedding layer cannot be used
        # and the input dimension will be the input feature size
        self.node_feature = node_feature
        if not node_feature:
            # if node_feature is not specified, node id will be used
            in_feats = 1

        super().__init__(in_feats, hidden_size, num_layers)

    def get_inputs(self, input_nodes, blocks):
        input_block = blocks[0]
        if self.node_feature:
            x = input_block.srcdata[self.node_feature]
        else:
            # It is the same as use input_nodes
            # It is not the same as the original dgl.NID if converted from heterogeneous graph
            # for example, the original ndata[dgl.NID] converted from heterogeneous graph
            # is [0, 1, 2, 0, 1, 2] while srcdata[dgl.NID] here
            # is [0, 1, 2, 3, 4, 5] which is the converted id
            x = input_block.srcdata[dgl.NID]
            # reshape from 1D int tensor to 2D float tensor
            x = x.reshape((x.size(dim=0), 1)).float()
        return x

    def get_inference_inputs(self, g):
        # base on the node feature, we can return all nodes feature tensor
        # or the node id tensor
        # None indicate it will get from the block directly
        return None

    def inference(self, g, x, device, batch_size):
        """Layer-wise inference algorithm to compute node embeddings.
        for the nodes of the entire graph.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set feature or None if use node id
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
                        h = self.get_inputs(input_nodes, blocks)
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
        # first iterating over the mini batches
        with dataloader.enable_cpu_affinity():
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
            ):
                # Inductive: use node feature or id
                h = self.get_inputs(input_nodes, blocks)
                for l, layer in enumerate(self.encoder.layers):
                    h = layer(blocks[l], h)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)

                # store the final output
                num_output = output_nodes.size(dim=0)
                y[y_start:y_start+num_output] = h.to(buffer_device)
                y_start += num_output

        return y

