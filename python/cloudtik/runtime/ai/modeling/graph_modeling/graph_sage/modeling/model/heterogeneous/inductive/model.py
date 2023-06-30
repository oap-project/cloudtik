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

import dgl

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.model import GraphSAGEModel


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

    def get_inputs(self, g, input_nodes, blocks):
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
