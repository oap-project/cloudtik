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

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.model import GraphSAGEModel


class TransductiveGraphSAGEModel(GraphSAGEModel):
    def __init__(self, vocab_size, hidden_size, num_layers, relations):
        super().__init__(hidden_size, hidden_size, num_layers, relations)

        # node embedding
        # vocab_size is a dict of vocab_size of each node type in the relations
        self.emb = torch.nn.ModuleDict(
            {node_type: torch.nn.Embedding(
                node_vocab_size, hidden_size) for node_type, node_vocab_size in vocab_size.items()})

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = {k: self.emb[k](v) for k, v in x.items()}
        return super().forward(
            pair_graph, neg_pair_graph, blocks, h)

    def get_input_embeddings(self):
        return {k: v.weight.data for k, v in self.emb.items()}

    def get_inputs(self, input_nodes, blocks):
        return input_nodes

    def get_inference_inputs(self, g):
        return self.get_input_embeddings()

    def get_encoder_inputs(self, input_nodes, blocks):
        x = self.get_inputs(input_nodes, blocks)
        return {k: self.emb[k].weight.data[v] for k, v in x.items()}
