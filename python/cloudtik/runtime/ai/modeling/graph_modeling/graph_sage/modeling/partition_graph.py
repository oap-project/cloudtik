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


import time
import os
import torch
import dgl


def partition_graph(
        dataset_dir, output_dir, graph_name,
        num_parts, num_hops):
    print("Random seed used in partitioning")
    dgl.random.seed(1)

    # create directories to save the partitions
    os.makedirs(output_dir, exist_ok=True)

    # load and preprocess dataset
    print("Loading data")
    start = time.time()
    # set force_reload=False if no changes on input graph (much faster otherwise ingestion ~30min)
    dataset = dgl.data.CSVDataset(dataset_dir, force_reload=False)
    print("time to load dataset from CSVs:", time.time() - start)

    hg = dataset[0]  # only one graph
    print(hg)
    print("etype to read train/test/val from:", hg.canonical_etypes[0][1])

    E = hg.num_edges(hg.canonical_etypes[0][1])
    reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])
    print("First reverse id is:  ", reverse_eids[0])

    # convert graph to homogeneous
    #TODO: handle new node ids for different node types
    g = dgl.to_homogeneous(hg)
    print(g)

    # part_method='random' works
    # part_method='metis' is giving a "free(): corrupted unsorted chunks error"
    # with multigraph (multiple links between same pair of nodes)
    # part_method='metis' works if you first do "dgl.to_simple(hg)" to keep single edge between pair of nodes
    # but that is not appropriate since we want to keep all multigraph edges

    nmap, emap = dgl.distributed.partition_graph(
        g,
        graph_name,
        num_parts,
        num_hops=num_hops,
        part_method="random",
        out_path=output_dir,
        balance_edges=True,
        return_mapping=True,
    )

    torch.save(nmap, os.path.join(output_dir, "nmap.pt"))
    torch.save(emap, os.path.join(output_dir, "emap.pt"))

    # Load first partition to verify
    (
        g,
        node_feats,
        edge_feats,
        gpb,
        graph_name,
        ntypes_list,
        etypes_list,
    ) = dgl.distributed.load_partition(
        os.path.join(output_dir, graph_name + ".json"), 0
    )
