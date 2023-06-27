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

import torch
import dgl

import tqdm

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.distributed.utils import \
    get_node_split_indices, dist_tensor_dict_new
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.model import GraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import get_num_nodes_of_graph, tensor_dict_collect, \
    tensor_dict_to, tensor_dict_truncate, tensor_dict_store


class DistGraphSAGEModel(GraphSAGEModel):
    def __init__(self, in_feats, hidden_size, num_layers, relations):
        super().__init__(in_feats, hidden_size, num_layers, relations)

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
        buffer_device = torch.device("cpu")
        nodes = get_node_split_indices(g, self.relations)
        num_nodes = get_num_nodes_of_graph(g, self.relations)
        y = dist_tensor_dict_new(g, num_nodes, self.encoder.hidden_size, "h")

        for i, layer in enumerate(self.encoder.layers):
            if i == len(self.encoder.layers) - 1:
                y = dist_tensor_dict_new(g, num_nodes, self.encoder.out_feats, "h_last")

            print(f"|V|={num_nodes}, eval batch size: {batch_size}")

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
                # Heterogeneous: everything is a dict here
                # input_nodes is a dict, output_nodes is dict
                # input_block.srcdata[dgl.NID] is a dict
                if x is not None:
                    h = tensor_dict_collect(x, input_nodes)
                else:
                    # The first layer will come to here if x is None
                    h = self.get_encoder_inputs(input_nodes, blocks)
                h = tensor_dict_to(h, device)
                size = {k: block.number_of_dst_nodes(k) for k, v in h.items()}
                h_dst = tensor_dict_truncate(h, size)
                h = layer(block, (h, h_dst))
                if i != len(self.encoder.layers) - 1:
                    # Heterogeneous
                    h = {k: self.encoder.activation(v) for k, v in h.items()}

                tensor_dict_store(
                    h, y, output_nodes, buffer_device)
            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield
