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


def parse_reverse_edges(reverse_edges_str):
    reverse_edge_dict = {}
    reverse_edges = [x.strip() for x in reverse_edges_str.split(",")]
    for reverse_edge in reverse_edges:
        reverse_edge_parts = [x.strip() for x in reverse_edge.split(":")]
        if len(reverse_edge_parts) != 2:
            raise ValueError(
                "Invalid reverse edge specification. Format: edge_type:reverse_edge_type")
        reverse_edge_dict[reverse_edge_parts[0]] = reverse_edge_parts[1]
    return reverse_edge_dict


def exclude_reverse_edge_types(etypes, reverse_etypes):
    if not reverse_etypes:
        return set(etypes)

    valid = set()
    exclude = set()
    for edge_type in etypes:
        if edge_type in exclude:
            continue
        valid.add(edge_type)
        reverse_edge_type = reverse_etypes.get(edge_type)
        if reverse_edge_type:
            exclude.add(reverse_edge_type)
    return valid
