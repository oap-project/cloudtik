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
    heterogeneous.distributed.trainer import Trainer
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.inductive.distributed.model import DistInductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.transductive.distributed.model import DistTransductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.utils import get_node_types
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.utils \
    import get_in_feats_of_feature


def main(args):
    seed = 7
    print("random seed set to:", seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    # load the partitioned heterogeneous graph
    print("Load partitioned graph")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        torch.distributed.init_process_group(backend="gloo")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)

    # Heterogeneous: create node indices based on the relations
    if args.relations:
        args.relations = [relation for relation in args.relations.split(",")]
    else:
        args.relations = g.etypes

    # create model here
    if args.inductive:
        feature_str = args.node_feature if args.node_feature else "ID"
        print("Training an inductive model on heterogeneous graph with feature:", feature_str)
        in_feats = get_in_feats_of_feature(g, args.node_feature)
        model = DistInductiveGraphSAGEModel(
            in_feats, args.num_hidden, args.num_layers,
            relations=args.relations, node_feature=args.node_feature)
        model_eval = DistInductiveGraphSAGEModel(
            in_feats, args.num_hidden, args.num_layers,
            relations=args.relations, node_feature=args.node_feature)
    else:
        print("Training a transductive model on heterogeneous graph")
        # vocab_size is a dict of node type as key
        node_types = get_node_types(g, args.relations)
        vocab_size = {k: g.num_nodes(k) for k in node_types}
        model = DistTransductiveGraphSAGEModel(
            vocab_size, args.num_hidden, args.num_layers,
            relations=args.relations)
        model_eval = DistTransductiveGraphSAGEModel(
            vocab_size, args.num_hidden, args.num_layers,
            relations=args.relations)

    trainer = Trainer(model, model_eval, args)
    trainer.train(g)


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

    # Heterogeneous
    parser.add_argument(
        "--relations",
        type=str,
        help="The comma separated list of edge relations for the heterogeneous model.")

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
