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


def get_effective_edge_types(g, reverse_etypes):
    return exclude_reverse_edge_types(g.etypes, reverse_etypes)


def get_eids_mask(
        g, mask_name, reverse_etypes=None, full_padded=False):
    # we include all the edges or first half of the reverse edges
    # we need to padding the spaces if there are more than one pair of reverse etypes
    effective = get_effective_edge_types(g, reverse_etypes)
    mask_list = []
    padding_num_edges = 0
    for etype in g.canonical_etypes:
        edge_type = etype[1]
        if edge_type not in effective:
            padding_num_edges += g.num_edges(edge_type)
        else:
            if padding_num_edges > 0:
                # padding zeros
                mask = torch.zeros((padding_num_edges,), dtype=torch.bool)
                mask_list.append(mask)
                # clear the padding list
                padding_num_edges = 0
            # append mask after the padding
            mask = g.edges[edge_type].data[mask_name]
            mask_list.append(mask)

    # handle the final padding if full padded is needed
    if full_padded and padding_num_edges > 0:
        # padding zeros
        mask = torch.zeros((padding_num_edges,), dtype=torch.bool)
        mask_list.append(mask)

    if len(mask_list) == 1:
        full_mask = mask_list[0]
    else:
        full_mask = torch.cat(mask_list)
    return full_mask


def get_eids_mask_full_padded(
        g, mask_name, reverse_etypes=None):
    # Optimize implementation of a full padded mask
    num_edges = g.num_edges()
    # we include all the edges or first half of the reverse edges
    # we need to padding the spaces if there are more than one pair of reverse etypes
    effective = get_effective_edge_types(g, reverse_etypes)
    # the full mask will all the edges
    full_mask = torch.zeros((num_edges,), dtype=torch.bool)
    base_idx = 0
    for etype in g.canonical_etypes:
        edge_type = etype[1]
        n = g.num_edges(edge_type)
        if edge_type not in effective:
            pass
        else:
            mask = g.edges[edge_type].data[mask_name]
            full_mask[base_idx: base_idx + n] = mask
        base_idx += n
    return full_mask


def get_eids_from_mask(
        g, mask_name, reverse_etypes=None, full_padded=False):
    mask = get_eids_mask(g, mask_name, reverse_etypes, full_padded)
    return torch.nonzero(mask, as_tuple=False).squeeze()


def _create_edge_prediction_sampler(sampler, reverse_eids):
    exclude = "reverse_id" if reverse_eids is not None else None
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude=exclude,
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )
    return sampler


def get_base_idx_map(graph):
    base_idx_map = {}
    base_idx = 0
    for etype in graph.canonical_etypes:
        edge_name = etype[1]
        base_idx_map[edge_name] = base_idx
        base_idx += graph.num_edges(edge_name)
    return base_idx_map


def get_reverse_eids(graph, reverse_etypes):
    # generate a base index map
    base_idx_map = get_base_idx_map(graph)
    reverse_eids_list = []
    for etype in graph.canonical_etypes:
        edge_name = etype[1]
        reverse_edge_name = reverse_etypes.get(edge_name)
        if not reverse_edge_name:
            raise ValueError(
                "Reverse edges not specified for {}".format(edge_name))
        num_edges = graph.num_edges(edge_name)
        if num_edges != graph.num_edges(reverse_edge_name):
            raise RuntimeError("The number of edges are not identical: {} <-> {}".format(
                edge_name, reverse_edge_name))
        base_idx = base_idx_map[reverse_edge_name]
        reverse_eids_list += [torch.arange(base_idx, base_idx + num_edges)]
    reverse_eids = torch.cat(reverse_eids_list)
    return reverse_eids
