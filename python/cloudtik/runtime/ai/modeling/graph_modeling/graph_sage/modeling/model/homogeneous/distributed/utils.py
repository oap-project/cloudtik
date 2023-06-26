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


def get_mask_padded(g, mask_name, num_pad):
    etype = g.canonical_etypes[0]
    edge_name = etype[1]
    mask = g.edges[edge_name].data[mask_name]
    mask_padded = torch.zeros((num_pad,), dtype=torch.bool)
    mask_padded[: mask.shape[0]] = mask
    return mask_padded


def get_edix_from_mask(g, mask_name, num_pad, mapping):
    mask_padded = get_mask_padded(g, mask_name, num_pad)
    shuffled_mask = torch.zeros((num_pad,), dtype=torch.bool)
    shuffled_mask[mapping] = mask_padded
    return torch.nonzero(shuffled_mask, as_tuple=False).squeeze()
