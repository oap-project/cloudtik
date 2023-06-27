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

import argparse
import random

import dgl
import numpy as np
import torch

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.distributed.trainer import Trainer
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.transductive.distributed.model import DistTransductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.inductive.distributed.model import DistInductiveGraphSAGEModel


def main(args):
    seed = 7
    print("random seed set to:", seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    # load original full graph to get the train/test/val id sets
    print("Loading original data to get the global train/test/val masks")
    dataset = dgl.data.CSVDataset(args.dataset_dir, force_reload=False)

    # Only one graph
    graph = dataset[0]
    print(graph)

    # create model here
    if args.inductive:
        print("Training an inductive model on homogeneous graph")
        in_feats = 1
        if args.node_feature:
            in_feats = graph.ndata[args.node_feature].shape[1]
        model = DistInductiveGraphSAGEModel(
            in_feats, args.num_hidden, args.num_layers,
            node_feature=args.node_feature)
        model_eval = DistInductiveGraphSAGEModel(
            in_feats, args.num_hidden, args.num_layers,
            node_feature=args.node_feature)
    else:
        vocab_size = graph.num_nodes()
        # use two models, one for distributed training, one for local evaluation
        model = DistTransductiveGraphSAGEModel(vocab_size, args.num_hidden, args.num_layers)
        model_eval = DistTransductiveGraphSAGEModel(vocab_size, args.num_hidden, args.num_layers)

    trainer = Trainer(model, model_eval, args)
    trainer.train(graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph SAGE Distributed")
    parser.add_argument(
        "--dataset-dir", "--dataset_dir",
        type=str,
        help="input dir with CSVDataset files"
    )
    parser.add_argument(
        "--model-file", "--model_file",
        type=str,
        help="output for model /your_path/model_graphsage_2L_64.pt",
    )
    parser.add_argument(
        "--node-embeddings-file", "--node_embeddings_file",
        type=str,
        help="node embeddings output: /your_path/node_emb.pt",
    )

    parser.add_argument(
        "--graph-name", "--graph_name",
        type=str,
        help="graph name")
    parser.add_argument(
        "--id",
        type=int,
        help="the partition id")
    parser.add_argument(
        "--ip-config", "--ip_config",
        type=str,
        help="The file for IP configuration")
    parser.add_argument(
        "--part-config", "--part_config",
        type=str,
        help="The path to the partition config file"
    )
    parser.add_argument(
        "--n-classes", "--n_classes",
        type=int,
        help="the number of classes")
    parser.add_argument(
        "--num-gpus", "--num_gpus",
        type=int, default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )

    parser.add_argument("--num-epochs", "--num_epochs",
                        type=int, default=3)
    parser.add_argument("--num-hidden", "--num_hidden",
                        type=int, default=16)
    parser.add_argument("--num-layers", "--num_layers",
                        type=int, default=2)
    parser.add_argument("--fan-out", "--fan_out",
                        type=str, default="10,25")
    parser.add_argument("--batch-size", "--batch_size",
                        type=int, default=1000)
    parser.add_argument("--batch-size-eval", "--batch_size_eval",
                        type=int, default=100000)
    parser.add_argument("--eval-every", "--eval_every",
                        type=int, default=5)
    parser.add_argument("--lr",
                        type=float, default=0.003)

    parser.add_argument("--log-every", "--log_every",
                        type=int, default=20)

    parser.add_argument(
        "--local-rank", "--local_rank",
        type=int,
        help="get rank of the process")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="run in the standalone mode"
    )
    parser.add_argument(
        "--num-negs", "--num_negs",
        type=int, default=1)
    parser.add_argument(
        "--neg-share", "--neg_share",
        default=False, action="store_true",
        help="sharing neg nodes for positive nodes",
    )
    parser.add_argument(
        "--dgl-sparse", "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument(
        "--sparse-lr", "--sparse_lr",
        type=float, default=1e-2,
        help="sparse lr rate")

    # Reverse edges to exclude during training
    parser.add_argument(
        "--reverse-edges", "--reverse_edges",
        type=str,
        help="The comma separated list of reverse edges mappings if has. "
             "For example, follow:follow-by,follow-by:follow,")
    parser.add_argument(
        "--exclude-reverse-edges", "--exclude_reverse_edges",
        default=False, action="store_true",
        help="whether to exclude reverse edges during sampling",
    )

    # Inductive
    parser.add_argument(
        "--inductive",
        action="store_true", default=False,
        help="Train an inductive model"
    )
    parser.add_argument(
        "--node-feature", "--node_feature",
        type=str,
        help="The feature name to use for node. If not set, will use node id.")

    args = parser.parse_args()
    print(args)

    main(args)
