# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import argparse
import random
import time

import dgl
import numpy as np
import torch

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.transductive.trainer import Trainer


def main(args):
    # random seeds for testing
    seed = 7

    print("Random seed set to: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # load and preprocess dataset
    print("Loading data")
    start = time.time()
    # set force_reload=False if no changes on input graph (much faster otherwise ingestion ~30min)
    dataset = dgl.data.CSVDataset(args.dataset_dir, force_reload=False)
    print("Time to load data from CSVs: ", time.time() - start)

    trainer = Trainer(args)
    model = trainer.train(dataset, device)
    graph = trainer.graph

    print("Inference to generate node representations...")
    model.eval()
    model.load_state_dict(torch.load(args.model_file))

    # Since it is transductive, the entire embedding includes all nodes
    x = model.get_input_embeddings()
    node_emb = model.inference(
        graph, x, device, args.batch_size_eval)
    print("Node embeddings shape: ", node_emb.shape)
    torch.save(node_emb, args.node_embeddings_file)

    # roc_auc on the different splits after training completion
    # roc_auc_test = evaluate(model, test_dataloader)
    # print("final test roc_auc", roc_auc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph SAGE")
    parser.add_argument(
        "--dataset-dir", "--dataset_dir",
        type=str,
        help="Path to CSVDataset")
    parser.add_argument(
        "--model-file", "--model_file",
        type=str, default="model_graphsage_2L_64.pt",
        help="output for model /your_path/model_graphsage_2L_64.pt")
    parser.add_argument(
         "--node-embeddings-file", "--node_embeddings_file",
         type=str, default="node_emb.pt",
         help="node emb output: /your_path/node_emb.pt")

    parser.add_argument(
        "--mode",
        default="cpu", choices=["cpu", "gpu", "mixed"],
        help="Training mode. 'cpu' for CPU training, 'gpu' for pure-GPU training, "
             "'mixed' for CPU-GPU mixed training.")

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

    parser.add_argument("--num-dl-workers", "--num_dl_workers",
                        type=int, default=4)

    args = parser.parse_args()
    print(args)

    main(args)
