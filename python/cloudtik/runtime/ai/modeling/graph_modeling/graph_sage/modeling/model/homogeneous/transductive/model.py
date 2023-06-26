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

    def get_encoder_inputs(self, input_nodes, blocks):
        x = self.get_inputs(input_nodes, blocks)
        return self.emb.weight.data[x]
