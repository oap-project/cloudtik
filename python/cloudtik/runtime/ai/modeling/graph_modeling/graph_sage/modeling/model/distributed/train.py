# Modifications Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import argparse
import os
import random

import dgl
import numpy as np
import torch as th


def main(args):
    seed = 7
    print("random seed set to: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    th.random.manual_seed(seed)
    th.manual_seed(seed)

    # load original full graph to get the train/test/val id sets
    print("Loading original data to get the global train/test/val masks")
    dataset = dgl.data.CSVDataset(args.dataset_dir, force_reload=False)

    hg = dataset[0]  # only one graph
    print(hg)
    print("etype to read train/test/val from: ", hg.canonical_etypes[0][1])
    train_mask = hg.edges[hg.canonical_etypes[0][1]].data["train_mask"]
    val_mask = hg.edges[hg.canonical_etypes[0][1]].data["val_mask"]
    test_mask = hg.edges[hg.canonical_etypes[0][1]].data["test_mask"]

    E = hg.num_edges(hg.canonical_etypes[0][1])
    # The i-th element indicates the ID of the i-th edge’s reverse edge.
    rev_eids_map = th.cat([th.arange(E, 2 * E), th.arange(0, E)])

    # load the partitioned graph (from homogeneous)
    print("Load prepartitioned graph")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend="gloo")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)

    train_mask_padded = th.zeros((g.num_edges(),), dtype=th.bool)
    train_mask_padded[: train_mask.shape[0]] = train_mask
    val_mask_padded = th.zeros((g.num_edges(),), dtype=th.bool)
    val_mask_padded[: val_mask.shape[0]] = val_mask
    test_mask_padded = th.zeros((g.num_edges(),), dtype=th.bool)
    test_mask_padded[: test_mask.shape[0]] = test_mask

    print("shuffle edges according to emap from global")
    shuffled_val_mask = th.zeros((g.num_edges(),), dtype=th.bool)
    shuffled_test_mask = th.zeros((g.num_edges(),), dtype=th.bool)
    shuffled_rev_eids_map = th.zeros_like(rev_eids_map)
    emap = th.load(os.path.join(os.path.dirname(args.part_config), "emap.pt"))
    print("emap shape: ", emap.shape)
    shuffled_val_mask[emap] = val_mask_padded
    shuffled_val_eidx = th.nonzero(shuffled_val_mask, as_tuple=False).squeeze()
    shuffled_test_mask[emap] = test_mask_padded
    shuffled_test_eidx = th.nonzero(shuffled_test_mask, as_tuple=False).squeeze()
    shuffled_rev_eids_map[emap] = rev_eids_map

    train_eids = dgl.distributed.edge_split(
        train_mask_padded,
        g.get_partition_book(),
        force_even=True,
    )

    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))

    # Pack data
    data = (
        train_eids,
        shuffled_val_eidx,
        shuffled_test_eidx,
        shuffled_rev_eids_map,
        g,
    )
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph SAGE Distributed")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration")
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)

    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--local_rank", type=int, help="get rank of the process")
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument("--num_negs", type=int, default=1)
    parser.add_argument(
        "--neg_share",
        default=False,
        action="store_true",
        help="sharing neg nodes for positive nodes",
    )
    parser.add_argument(
        "--remove_edge",
        default=False,
        action="store_true",
        help="whether to remove edges during sampling",
    )
    parser.add_argument(
        "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument("--sparse_lr", type=float, default=1e-2, help="sparse lr rate")
    parser.add_argument(
        "--dataset_dir", type=str, help="input dir with CSVDataset files"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="output for model /your_path/model_graphsage_2L_64.pt",
    )
    parser.add_argument(
        "--node_embeddings_file",
        type=str,
        help="node embeddings output: /your_path/node_emb.pt",
    )
    args = parser.parse_args()
    print(args)

    main(args)
