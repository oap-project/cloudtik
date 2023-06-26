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
    heterogeneous.model import GraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import get_node_indices, get_num_nodes_of_graph, get_total_num_nodes, \
    get_num_nodes_of_nids, tensor_dict_new, tensor_dict_to, tensor_dict_store, \
    tensor_dict_collect, tensor_dict_store_at


class InductiveGraphSAGEModel(GraphSAGEModel):
    def __init__(self, in_feats, hidden_size, num_layers,
                 relations, node_feature=None):

        # Inductive: embedding layer cannot be used
        # and the input dimension will be the input feature size
        self.node_feature = node_feature
        if not node_feature:
            # if node_feature is not specified, node id will be used
            in_feats = 1

        super().__init__(in_feats, hidden_size, num_layers, relations)

    def get_inputs(self, input_nodes, blocks):
        input_block = blocks[0]
        if self.node_feature:
            x = input_block.srcdata[self.node_feature]
        else:
            x = input_block.srcdata[dgl.NID]
            # reshape from 1D int tensor to 2D float tensor
            x = {k: v.reshape((v.size(dim=0), 1)).float() for k, v in x.items()}
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
                    # Inductive: use node feature or id
                    # Heterogeneous: everything is a dict here
                    # input_nodes is a dict, output_nodes is dict
                    # input_block.srcdata[dgl.NID] is a dict
                    if x is not None:
                        h = tensor_dict_collect(x, input_nodes)
                    else:
                        # The first layer will come to here if x is None
                        h = self.get_inputs(input_nodes, blocks)
                    tensor_dict_to(h, device)
                    h = layer(blocks[0], h)
                    if l != len(self.encoder.layers) - 1:
                        h = {k: F.relu(v) for k, v in h.items()}

                    tensor_dict_store(
                        h, y, output_nodes, buffer_device)
                x = y
        return y

    def inference_nodes(self, g, nids, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings.
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
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
                h = self.get_inputs(input_nodes, blocks)
                tensor_dict_to(h, device)

                for l, layer in enumerate(self.encoder.layers):
                    h = layer(blocks[l], h)
                    if l != len(self.encoder.layers) - 1:
                        h = {k: F.relu(v) for k, v in h.items()}

                # store the final output
                tensor_dict_store_at(
                    h, y, output_nodes, y_start, buffer_device)

        return y
