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

import numpy as np
import torch
import dgl

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import get_node_types, get_effective_edge_types, tensor_dict_shape


def get_node_split_indices(g, relations):
    node_types = get_node_types(g, relations)

    # Note that the parameter to node_split is:
    # A boolean mask vector that indicates input nodes.
    return {node_type: dgl.distributed.node_split(
        torch.ones(g.num_nodes(node_type), dtype=torch.bool),
        g.get_partition_book(),
        ntype=node_type,
        force_even=True,
    ) for node_type in node_types}


def get_edge_split_indices(g, relations, edge_mask, reverse_etypes):
    effective = get_effective_edge_types(relations, reverse_etypes)
    return {edge_type: dgl.distributed.edge_split(
        edge_mask[edge_type],
        g.get_partition_book(),
        etype=edge_type,
        force_even=True,
    ) for edge_type in relations if edge_type in effective}


def get_eids_mask(g, relations, mask_name, reverse_etypes=None):
    mask_dict = {}
    effective = get_effective_edge_types(relations, reverse_etypes)
    for edge_type in relations:
        if edge_type not in effective:
            continue
        mask_dict[edge_type] = g.edges[edge_type].data[mask_name]
    return mask_dict


def get_eids_from_mask(g, relations, mask_name, mapping, reverse_etypes=None):
    eids_dict = {}
    effective = get_effective_edge_types(relations, reverse_etypes)
    for edge_type in relations:
        if edge_type not in effective:
            continue
        # the mapping uses canonical_etype as key
        canonical_etype = g.to_canonical_etype(edge_type)
        edge_mapping = mapping[canonical_etype]

        mask = g.edges[edge_type].data[mask_name]
        shuffled_mask = torch.zeros(mask.shape, dtype=mask.dtype)
        shuffled_mask[edge_mapping] = mask
        eids_dict[edge_type] = torch.nonzero(
            shuffled_mask, as_tuple=False).squeeze()
    return eids_dict


def save_node_embeddings(node_emb, output_file):
    # node_emb is dict with DistTensor, convert to a torch Tensor by copying
    local_node_emb = {k: v[0: v.shape[0]] for k, v in node_emb.items()}
    torch.save(local_node_emb, output_file)
    print("Node embeddings shape:", tensor_dict_shape(local_node_emb))
    print("Saved node embeddings to:", output_file)


def dist_tensor_dict_new(
        g, num_nodes, out_size,
        name_suffix):
    y = {}
    for ntype, num in num_nodes.items():
        y[ntype] = dgl.distributed.DistTensor(
            (num, out_size),
            torch.float32,
            ntype + name_suffix,
            part_policy=g.get_node_partition_policy(ntype),
            persistent=True,
        )
    return y
