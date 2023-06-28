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

import pandas as pd
from collections import OrderedDict


def column_index(series, offset=0):
    return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}


def get_node_type_columns(node_columns):
    node_type_columns = {}
    for column, node_type in node_columns:
        if node_type not in node_type_columns:
            node_type_columns[node_type] = []
        node_type_columns[node_type].append(column)
    return node_type_columns


def values_of_node(df, columns):
    node_value_list = []
    for column in columns:
        column_values = df[column]
        node_value_list.append(column_values)
    if len(node_value_list) == 1:
        node_values = node_value_list[0]
    else:
        node_values = pd.concat(node_value_list, ignore_index=True)
    return node_values


def get_mapped_column_of(column_name, col_map, config):
    node_columns = config["node_columns"]
    # get node type of node column name
    node_type = node_columns[column_name]
    col_map_of_node = col_map[node_type]
    return col_map_of_node[column_name]


def mapping_of_node(df, columns, offset):
    node_values = values_of_node(df, columns)
    return column_index(node_values, offset=offset)


def map_columns(df, columns, mapping):
    col_map_of_node = {}
    for column in columns:
        new_col_name = column + "_idx"
        col_map_of_node[column] = new_col_name
        # add new Idx to dataframe
        df[new_col_name] = df[column].map(mapping)
    return col_map_of_node


def tokenize_node_ids(df, config, heterogeneous):
    # create dictionary of dictionary to store node mapping for all node types
    offset = 0
    mapping = OrderedDict()
    # create mapping dictionary between original IDs and incremental IDs starting at zero
    # Note: because GNN is converting the graph to homogeneous we need the homogeneous mapping here
    # i,e: node_0: [0, x] node_1: [x,y] node_2: [y,z]
    # each unique real node id will be assigned a indexed id.
    # all the node types indexed ids are fatten numbered in the order defined in "node_columns"
    # key in dict: node + "_2idx" stores a unique mapping from real id -> indexed id  mapping
    # column new_col_name: node + "_idx" stores the indexed id
    col_map = {}
    node_types = config["node_types"]
    node_type_columns = get_node_type_columns(config["node_columns"])
    for i, node in enumerate(node_types):
        key = str(node + "_2idx")
        columns = node_type_columns[node]
        mapping[key] = mapping_of_node(df, columns, offset=offset)
        col_map[node] = map_columns(df, columns, mapping[key])
        if not heterogeneous:
            offset = len(mapping[key])
    print("Tokenize column map:", col_map)
    return mapping, col_map
