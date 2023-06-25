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

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.tokenizer import tokenize_node_ids


def build_graph(
        input_file, output_dir, dataset_name,
        tabular2graph):
    with open(tabular2graph, "r") as file:
        config = yaml.safe_load(file)

    output_dataset_dir = os.path.join(output_dir, dataset_name)
    print(output_dataset_dir)
    os.makedirs(output_dataset_dir, exist_ok=True)

    # 1. Load CSV file output of Classical ML edge featurization workflow
    print("Loading processed data")
    start = time.time()
    df = pd.read_csv(input_file)  # , nrows=10000)
    t_load_data = time.time()
    print("Time to load processed data", t_load_data - start)

    # 2. Renumbering - generating node/edge ids starting from zero
    print("Node renumbering")
    # heterogeneous mapping where all types start from zero
    mapping, col_map = tokenize_node_ids(df, config, homogeneous=False)

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
    for i, node in enumerate(col_map.keys()):
        list_of_n_dict.append(
            {"file_name": "nodes_" + str(i) + ".csv", "ntype": col_map[node]}
        )

    list_of_e_dict = []
    for i, edge_type in enumerate(config["edge_types"]):
        # replace node ids with the re-enumerated Idx
        edge_type[0] = col_map[edge_type[0]]
        edge_type[2] = col_map[edge_type[2]]
        print(edge_type)
        # keep edge type from tabular2graph.yaml and update node type to the new Idx ones
        list_of_e_dict.append(
            {"file_name": "edges_" + str(i) + ".csv", "etype": edge_type}
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

    # write edges_.csv files
    for i, edge_type in enumerate(meta_yaml["edge_data"]):
        print("\nWriting: ", edge_type["file_name"])
        edge_header = ["src_id", "dst_id"]  # minimum required
        edge_df_cols = [edge_type["etype"][0], edge_type["etype"][2]]
        if config["edge_label"]:
            edge_header.append("label")
            edge_df_cols.append(config["edge_label"])
        if config["edge_split"]:
            # it is required that split has 3 values (0,1,2) for train test val respectively
            edge_header.extend(["train_mask", "val_mask", "test_mask"])
            edge_df_cols.extend(["masks_0", "masks_1", "masks_2"])
            # edge_header.extend(["train_mask"])
            # edge_df_cols.extend(["masks_0"])
        if config["edge_features"]:
            edge_features = config["edge_features"]
            print("Features for edges: ", edge_features)
            data_columns = set(df.columns)
            feat_keys = [feature for feature in edge_features if feature in data_columns]
            if len(feat_keys) != len(edge_features):
                print("Valid features for edges: ", feat_keys)
            # Note: feat_as_str needs to be a string of comma separated values
            # enclosed in double quotes for dgl default parser to work
            df["edge_feat_as_str"] = df[feat_keys].astype(str).apply(",".join, axis=1)
            edge_header.append("feat")
            edge_df_cols.append("edge_feat_as_str")
        assert len(edge_df_cols) == len(edge_header)
        df[edge_df_cols].to_csv(
            os.path.join(output_dataset_dir, edge_type["file_name"]),
            index=False,
            header=edge_header,
        )
    # write nodes_.csv files
    for i, node in enumerate(meta_yaml["node_data"]):
        print("\nWriting: ", meta_yaml["node_data"][i]["file_name"])
        print(df[meta_yaml["node_data"][i]["ntype"]].unique())
        np.savetxt(
            os.path.join(output_dataset_dir, meta_yaml["node_data"][i]["file_name"]),
            df[meta_yaml["node_data"][i]["ntype"]].unique(),
            delimiter=",",
            header="node_id",
            comments="",
        )

    t_csv_dataset = time.time()
    print("Time to write CSVDatasets", t_csv_dataset - t_renum)
