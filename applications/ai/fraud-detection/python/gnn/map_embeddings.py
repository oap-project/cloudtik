# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import time
import argparse
import os
import yaml
from collections import OrderedDict
import torch
import pandas as pd

from .utils import existing_file
from .utils import existing_directory


def map_embeddings(
        processed_data_file, partition_dir,
        node_embeddings_dir, node_embeddings_name,
        output_file, tabular2graph):
    node_embeddings_file = node_embeddings_dir + "/" + node_embeddings_name + ".pt"
    mapped_node_embeddings_file = node_embeddings_dir + "/" + node_embeddings_name + "_mapped.pt"
    nmap_file = os.path.join(partition_dir, "nmap.pt")

    with open(tabular2graph, "r") as file:
        config = yaml.safe_load(file)

    # 1. Load CSV file output of Classical ML edge featurization workflow
    print("Loading processed data")
    start = time.time()
    df = pd.read_csv(processed_data_file)
    t_load_data = time.time()
    print("Time lo load processed data", t_load_data - start)

    # 2. Renumbering - generating node/edge ids starting from zero
    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    # create dictionary of dictionary to stare node mapping for all node types
    offset = 0
    dict = OrderedDict()
    # create mapping dictionary between original IDs and incremental IDs starting at zero
    # Note: because GNN is converting the graph to homogeneous we need the homogeneous mapping here
    # i,e: node_0: [0, x] node_1: [x,y] node_2: [y,z]
    col_map = {}
    for i, node in enumerate(config["node_columns"]):
        key = str(node + "_2idx")
        dict[key] = column_index(df[config["node_columns"][i]], offset=offset)
        new_col_name = node + "_Idx"
        col_map[node] = new_col_name
        # add new Idx to dataframe
        df[new_col_name] = df[config["node_columns"][i]].map(dict[key])
        offset = len(
            dict[key]
        )  # homogeneous mapping because that is how the embeddings will be returned bby GNN
    t_renum = time.time()
    print("Re-enumerated column map (homogeneous mapping): ", col_map)
    print("Time to renumerate", t_renum - t_load_data)

    # 3. Load node embeddings from file, add them to edge features
    # and save file for Classic ML workflow (since model is trained as homo, no mapping needed.)
    print("Loading embeddings from file and adding to preprocessed CSV file")
    if not os.path.isfile(node_embeddings_file):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(node_embeddings_file)
        )
    node_emb = torch.load(node_embeddings_file)

    # 4. Map from partition to global
    print("Mapping from partition ids to full graph ids")
    nmap = torch.load(nmap_file)

    orig_node_emb = torch.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[nmap] = node_emb
    torch.save(orig_node_emb, mapped_node_embeddings_file)

    node_emb_arr = orig_node_emb.cpu().detach().numpy()
    node_emb_dict = {i: val for i, val in enumerate(node_emb_arr)}
    t_load_emb = time.time()
    print("Time to load embeddings", t_load_emb - t_load_data)

    for i, node in enumerate(col_map.keys()):
        emb = pd.DataFrame(df[col_map[node]].map(node_emb_dict).tolist()).add_prefix(
            "n" + str(i) + "_e"
        )
        df = df.join([emb])
        df.drop(
            columns=[col_map[node]],
            axis=1,
            inplace=True,
        )

    print("CSV output shape: ", df.shape)

    df.to_csv(output_file, index=False)
    print(
        "Time to append node embeddings to edge features CSV", time.time() - t_load_emb)


def main(args):
    map_embeddings(
        processed_data_file=args.processed_data_file,
        partition_dir=args.partition_dir,
        node_embeddings_dir=args.node_embeddings_dir,
        node_embeddings_name=args.node_embeddings_name,
        output_file=args.output_file,
        tabular2graph=args.tabular2graph
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapEmb and save")
    parser.add_argument(
        "--processed_data_file", type=existing_file, help="The path to the processed data in csv"
    )
    parser.add_argument(
        "--partition_dir", type=existing_directory, help="The path to the partition folder"
    )
    parser.add_argument(
        "--node_embeddings_dir",
        type=existing_directory,
        help="The path to the pt files generated in training")
    parser.add_argument(
        "--node_embeddings_name",
        type=str,
        default="node_emb",
        help="The path to the node embedding file")
    parser.add_argument(
        "--tabular2graph",
        required=True,
        help="The path to the tabular2graph.yaml")
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to the csv data file with mapped node embeddings")
    args = parser.parse_args()
    print(args)

    main(args)
