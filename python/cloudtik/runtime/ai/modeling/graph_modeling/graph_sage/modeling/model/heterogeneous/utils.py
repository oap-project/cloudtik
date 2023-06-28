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

from dgl.dataloading import (
    as_edge_prediction_sampler,
    negative_sampler,
)

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.utils import \
    exclude_reverse_edge_types


def _create_edge_prediction_sampler(sampler, reverse_etypes):
    exclude = "reverse_types" if reverse_etypes is not None else None
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude=exclude,
        reverse_etypes=reverse_etypes,
        negative_sampler=negative_sampler.Uniform(1),
    )
    return sampler


def get_eids_from_mask(g, relations, mask_name, reverse_etypes):
    eids_dict = {}
    # we include only the eids from relations
    # we include all the edges or first half of the reverse edges
    effective = get_effective_edge_types(relations, reverse_etypes)
    for edge_type in relations:
        if edge_type not in effective:
            continue
        mask = g.edges[edge_type].data[mask_name]
        eids_dict[edge_type] = torch.nonzero(mask, as_tuple=False).squeeze()
    return eids_dict


# Most functions operating for heterogeneous graph are on dict of tensors

def get_node_types(g, relations):
    node_types = set()
    for etype in g.canonical_etypes:
        if etype[1] in relations:
            node_types.add(etype[0])
            node_types.add(etype[2])
    return node_types


def get_effective_edge_types(relations, reverse_etypes):
    return exclude_reverse_edge_types(relations, reverse_etypes)


def get_node_indices(g, relations):
    node_types = get_node_types(g, relations)
    return {node_type: torch.arange(
        g.num_nodes(node_type)).to(g.device) for node_type in node_types}


def get_num_nodes_of_graph(g, relations):
    node_types = get_node_types(g, relations)
    return {node_type: g.num_nodes(
        node_type) for node_type in node_types}


def get_num_nodes_of_nids(g, relations, nids):
    node_types = get_node_types(g, relations)
    return {node_type: ids.size(
        dim=0) for node_type, ids in nids.items() if node_type in node_types}


def get_total_num_nodes(nids):
    if not isinstance(nids, dict):
        return nids.size(dim=0)
    num_nodes = 0
    for ntype, ids in nids.items():
        num_nodes += ids.size(dim=0)
    return num_nodes


def tensor_dict_new(
        num_nodes, out_size,
        output_device, pin_memory):
    y = {}
    for ntype, num in num_nodes.items():
        y[ntype] = torch.empty(
            num,
            out_size,
            device=output_device,
            pin_memory=pin_memory,
        )
    return y


def tensor_dict_to(x, device):
    if not isinstance(x, dict):
        return x.to(device)
    return {k: v.to(device) for k, v in x.items()}


def tensor_dict_store(x, y, indices, device):
    if not isinstance(x, dict):
        y[indices] = x.to(device)
        return

    for k, v in x.items():
        i = indices[k]
        y_v = y[k]
        y_v[i] = v.to(device)


def tensor_dict_store_at(x, y, indices, pos, device):
    if not isinstance(x, dict):
        y[pos: pos + indices.size(dim=0)] = x.to(device)
        return

    for k, v in x.items():
        start = pos[k]
        end = start + indices[k].size(dim=0)
        y_v = y[k]
        y_v[start: end] = v.to(device)
        pos[k] = end


def tensor_dict_collect(x, indices):
    if not isinstance(x, dict):
        return x[indices]
    return {k: x[k][i] for k, i in indices.items()}


def tensor_dict_flatten(x):
    if not isinstance(x, dict):
        return x
    # we should the sort key, so that it is predictable
    return torch.cat([v for k, v in sorted(x.items())])


def tensor_dict_shape(x):
    if not isinstance(x, dict):
        return x.shape
    return {k: v.shape for k, v in x.items()}


def tensor_dict_truncate(x, size):
    if not isinstance(x, dict):
        return x[: size]
    return {k: v[: size[k]] for k, v in x.items()}
