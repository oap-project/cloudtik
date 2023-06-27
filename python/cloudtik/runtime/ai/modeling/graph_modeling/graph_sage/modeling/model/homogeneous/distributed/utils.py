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

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.homogeneous.utils import \
    get_eids_mask_full_padded


def get_eids_from_mask(g, mask_name, mapping, reverse_etypes=None):
    num_edges = g.num_edges()
    mask_padded = get_eids_mask_full_padded(g, mask_name, reverse_etypes)
    shuffled_mask = torch.zeros((num_edges,), dtype=torch.bool)
    shuffled_mask[mapping] = mask_padded
    return torch.nonzero(shuffled_mask, as_tuple=False).squeeze()


def save_node_embeddings(node_emb, output_file):
    # node_emb is DistTensor, convert to a torch Tensor by copying
    local_node_emb = node_emb[0: node_emb.shape[0]]
    torch.save(local_node_emb, output_file)
    print("Node embeddings shape:", local_node_emb.shape)
    print("Saved node embeddings to:", output_file)
