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
import numpy as np

import time
import yaml
import os

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.tokenizer import tokenize_node_ids, \
    get_node_type_columns, values_of_node, get_mapped_column_of, get_node_type_of_column
from cloudtik.runtime.ai.util.utils import clean_dir


def build_graph(
        input_file, output_dir, dataset_name,
        tabular2graph):
    with open(tabular2graph, "r") as file:
        config = yaml.safe_load(file)

    node_types = config["node_types"]
    node_columns = config["node_columns"]
    node_features = config.get("node_features")
    edge_types = config["edge_types"]
    edge_features = config.get("edge_features")

    print("Build graph for:")
    print("    node types:", node_types)
    print("    edge types:", edge_types)

    output_dataset_dir = os.path.join(output_dir, dataset_name)
    print(output_dataset_dir)
    clean_dir(output_dataset_dir)

    # 1. Load CSV file output for preprocessing
    print("Loading processed data")
    start = time.time()
    df = pd.read_csv(input_file)  # , nrows=10000)
    t_load_data = time.time()
    print("Time to load processed data", t_load_data - start)

    # 2. Renumbering - generating node/edge ids starting from zero
    print("Node renumbering")
    # heterogeneous mapping where all types start from zero
    mapping, col_map = tokenize_node_ids(
        df, config, heterogeneous=True)

    t_renum = time.time()
    print("Time to renumerate", t_renum - t_load_data)

    # 3. create masks for train, val and test splits (add new columns with masks)
    if config["edge_split"]:
        df = pd.concat(
            [
                df,
                pd.get_dummies(
                    df[config["edge_split"]].astype("category"), prefix="masks"
                ),
            ],
            axis=1,
        )

    # 4. Prepare CSVDataset files for DGL to ingest and create graph
    print("Writing data into set of CSV files (nodes/edges)")
    # The specs for yaml file content and node/edge CSV file please refer to:
    # https://docs.dgl.ai/en/1.0.x/guide/data-loadcsv.html#guide-data-pipeline-loadcsv

    # programmatically create meta.yaml expected by DGL from yaml config
    list_of_n_dict = []
    for i, node_type in enumerate(node_types):
        list_of_n_dict.append(
            {"file_name": "nodes_" + str(i) + ".csv", "ntype": node_type}
        )

    list_of_e_dict = []
    for i, edge_type in enumerate(edge_types):
        src_node_type = get_node_type_of_column(edge_type[0], node_columns)
        dst_node_type = get_node_type_of_column(edge_type[2], node_columns)
        etype = [src_node_type, edge_type[1], dst_node_type]
        list_of_e_dict.append(
            {"file_name": "edges_" + str(i) + ".csv", "etype": etype}
        )

    with open(os.path.join(output_dataset_dir, "meta.yaml"), "w") as f:
        meta_yaml = {
            "dataset_name": dataset_name,
            "node_data": list_of_n_dict,
            "edge_data": list_of_e_dict,
        }
        data = yaml.dump(meta_yaml, f, sort_keys=False, default_flow_style=False)

    # avoid to write and read immediately (if the file system points to a distributed file system)
    # with open(os.path.join(output_dataset_dir, "meta.yaml"), "r") as file:
    #     meta_yaml = yaml.safe_load(file)
    print("\nmeta_yaml: \n", meta_yaml)

    # DGL node/edge csv headers
    # edge_header = ["src_id", "dst_id", "label", "train_mask", "val_mask","test_mask","feat"]
    # node_header = ["node_id", "label", "train_mask", "val_mask","test_mask","feat"]

    if node_features is None:
        node_features = {}

    # write nodes_.csv files
    node_type_columns = get_node_type_columns(node_columns)
    for i, node_meta in enumerate(meta_yaml["node_data"]):
        node_type = node_meta["ntype"]
        file_name = node_meta["file_name"]
        print("\nWriting {} node: {}".format(node_type, file_name))

        col_map_of_node = col_map[node_type]
        columns = node_type_columns[node_type]

        # check the node features and its validation
        num_features = _get_num_features_of_node(columns, node_features)
        if num_features <= 0:
            # if there is no future columns, simple path
            mapped_columns = [col_map_of_node[column] for column in columns]
            node_values = values_of_node(df, mapped_columns)
            np.savetxt(
                os.path.join(output_dataset_dir, file_name),
                node_values.unique(),
                delimiter=",",
                header="node_id",
                comments="",
            )
        else:
            print("Features for node:", {column: node_features.get(column) for column in columns})

            node_header = ["node_id", "feat"]  # minimum required
            node_df_cols = ["node_id", "node_feat_as_str"]

            # get list of tuples, each tuple is (mapped_column, feature_columns)
            # all the columns of the same node type must have the same identical features if has
            mapped_columns = [(col_map_of_node[column], node_features.get(column)) for column in columns]
            # all the columns will renamed in the form of (node_id, feat_1, feat_2, ...) and concat vertically
            df_node_columns = _concat_columns_of_node(df, mapped_columns)

            # use the node_id to drop the duplicates
            df_unique = df_node_columns.drop_duplicates(subset=['node_id'])

            feat_keys = [f"feat_{i}" for i in range(num_features)]
            # Note: feat_as_str needs to be a string of comma separated values
            # enclosed in double quotes for dgl default parser to work
            df_unique["node_feat_as_str"] = df_unique[feat_keys].astype(float).astype(str).apply(",".join, axis=1)
            if len(feat_keys) == 1:
                # append a "," if there is only one value for forcing to double quotes
                df_unique["node_feat_as_str"] = df_unique["node_feat_as_str"] + ","

            assert len(node_df_cols) == len(node_header)
            df_unique[node_df_cols].to_csv(
                os.path.join(output_dataset_dir, file_name),
                index=False,
                header=node_header,
            )

    # write edges_.csv files
    for i, edge_meta in enumerate(meta_yaml["edge_data"]):
        etype = edge_meta["etype"]
        edge_type = etype[1]
        file_name = edge_meta["file_name"]
        print("\nWriting {} edge: {}".format(edge_type, file_name))

        edge_header = ["src_id", "dst_id"]  # minimum required
        src_id = get_mapped_column_of(edge_types[i][0], col_map, node_columns)
        dst_id = get_mapped_column_of(edge_types[i][2], col_map, node_columns)
        edge_df_cols = [src_id, dst_id]
        if config["edge_label"]:
            edge_header.append("label")
            edge_df_cols.append(config["edge_label"])
        if config["edge_split"]:
            # it is required that split has 3 values (0,1,2) for train test val respectively
            edge_header.extend(["train_mask", "val_mask", "test_mask"])
            edge_df_cols.extend(["masks_0", "masks_1", "masks_2"])
            # edge_header.extend(["train_mask"])
            # edge_df_cols.extend(["masks_0"])
        if edge_features and edge_type in edge_features:
            features = edge_features[edge_type]
            print("Features for edge:", features)
            data_columns = set(df.columns)
            feat_keys = [feature for feature in features if feature in data_columns]
            if len(feat_keys) != len(features):
                print("Valid features for edge:", feat_keys)
            # Note: feat_as_str needs to be a string of comma separated values
            # enclosed in double quotes for dgl default parser to work
            df["edge_feat_as_str"] = df[feat_keys].astype(float).astype(str).apply(",".join, axis=1)
            if len(feat_keys) == 1:
                # append a "," if there is only one value for forcing to double quotes
                df_unique["node_feat_as_str"] = df_unique["node_feat_as_str"] + ","
            edge_header.append("feat")
            edge_df_cols.append("edge_feat_as_str")
        assert len(edge_df_cols) == len(edge_header)
        df[edge_df_cols].to_csv(
            os.path.join(output_dataset_dir, file_name),
            index=False,
            header=edge_header,
        )

    t_csv_dataset = time.time()
    print("Time to write CSVDatasets", t_csv_dataset - t_renum)


def _get_num_features_of_node(columns, node_features):
    num_features = -1
    for column in columns:
        num = len(node_features.get(column, []))
        if num_features >= 0:
            if num != num_features:
                raise ValueError("Columns for the node type have different number of features.")
        else:
            num_features = num
    return num_features


def _concat_columns_of_node(df, mapped_columns):
    node_dfs = []
    for mapped_column in mapped_columns:
        id_column = mapped_column[0]
        feature_columns = mapped_column[1]
        # select the needed columns
        sel_columns = [id_column]
        sel_columns += feature_columns
        node_df = df[sel_columns]
        # rename the columns
        rename_map = {id_column: "node_id"}
        for i, feature_column in enumerate(feature_columns):
            rename_map[feature_column] = f"feat_{i}"
        node_df.rename(columns=rename_map, inplace=True)
        node_dfs.append(node_df)
    if len(node_dfs) == 1:
        df_node_columns = node_dfs[0]
    else:
        df_node_columns = pd.concat(node_dfs, ignore_index=True)
    return df_node_columns
