# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv
from dgl.nn import EdgePredictor
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    MultiLayerFullNeighborSampler,
    as_edge_prediction_sampler,
    negative_sampler,
)

import tqdm
import argparse
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import random


class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(
            SAGEConv(hidden_size, out_feats, aggregator_type)
        )  # activation None

    def forward(self, graphs, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, graphs)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h


class Model(nn.Module):
    def __init__(self, vocab_size, hid_size, n_layers):
        super().__init__()

        self.hid_size = hid_size

        # node embedding
        self.emb = torch.nn.Embedding(vocab_size, hid_size)
        # encoder is a 1-layer GraphSAGE model
        self.encoder = GraphSAGE(hid_size, hid_size, hid_size, n_layers, F.relu, "mean")
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        # cosine similarity with linear
        # self.predictor=dglnn.EdgePredictor('cos',hid_size,1)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        h = self.encoder(blocks, h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # h_pos = self.predictor(h[pos_src], h[pos_dst])
        # h_neg = self.predictor(h[neg_src], h[neg_dst])
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        # feat = g.ndata['feat']
        # use pretrained embedding as node features
        feat = self.emb.weight.data
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        # compure representations layer by layer
        for l, layer in enumerate(self.encoder.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                feat = y
        return y


def evaluate(model, test_dataloader):
    model.eval()
    score_all = torch.tensor(())
    labels_all = torch.tensor(())
    with test_dataloader.enable_cpu_affinity():
        with torch.no_grad():
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                tqdm.tqdm(test_dataloader)
            ):
                x = blocks[0].srcdata[dgl.NID]
                pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                score_all = torch.cat([score_all, score])
                labels_all = torch.cat([labels_all, labels])
            rocauc = roc_auc_score(labels_all, score_all)
    return rocauc


def train(args, device, g, train_dataloader, val_dataloader, test_dataloader, model):
    best_rocauc = 0
    best_model_path = args.model_file
    print("learning rate: ", args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    with train_dataloader.enable_cpu_affinity():
        for epoch in range(args.num_epochs):
            start = time.time()
            step_time = []
            model.train()
            total_loss = 0
            total_val_loss = 0.0
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                train_dataloader
            ):
                # x = blocks[0].srcdata[dgl.NID] #this is equal to input_nodes
                pos_score, neg_score = model(
                    pair_graph, neg_pair_graph, blocks, input_nodes
                )
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                step_t = time.time() - start
                step_time.append(step_t)
                start = time.time()
            # print("Epoch {:05d} | Train Loss {:.4f}".format(epoch, total_loss / (it + 1)))
            # torch.save(model.state_dict(), best_model_path)

            model.eval()

            with val_dataloader.enable_cpu_affinity():
                for it2, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                    val_dataloader
                ):
                    # x = blocks[0].srcdata[dgl.NID] this is same as input_nodes
                    pos_score, neg_score = model(
                        pair_graph, neg_pair_graph, blocks, input_nodes
                    )
                    score = torch.cat([pos_score, neg_score])
                    pos_label = torch.ones_like(pos_score)
                    neg_label = torch.zeros_like(neg_score)
                    labels = torch.cat([pos_label, neg_label])
                    val_loss = F.binary_cross_entropy_with_logits(score, labels)
                    total_val_loss += val_loss.item()
                print(
                    "Epoch {:05d} | Epoch time {:.4f} | Train Loss {:.4f} | Valid Loss {:.4f}".format(
                        epoch,
                        np.sum(step_time),
                        total_loss / (it + 1),
                        total_val_loss / (it2 + 1),
                    )
                )

            if (epoch % args.eval_every == 0 and epoch != 0) or (
                epoch == args.num_epochs - 1
            ):
                rocauc = evaluate(model, test_dataloader)
                # update best model if needed
                if best_rocauc < rocauc:
                    print("Updating best model")
                    best_rocauc = rocauc
                    torch.save(model.state_dict(), best_model_path)
                print("Epoch {:05d} | Test roc_auc_score {:.4f}".format(epoch, rocauc))


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

    # load and preprocess dataset
    print("Loading data")
    start = time.time()
    # set force_reload=False if no changes on input graph (much faster otherwise ingestion ~30min)
    dataset = dgl.data.CSVDataset(args.dataset_dir, force_reload=False)
    print("Time to load data from CSVs: ", time.time() - start)

    hg = dataset[0]  # only one graph
    print(hg)
    print("etype to read train/test/val from: ", hg.canonical_etypes[0][1])
    train_mask = hg.edges[hg.canonical_etypes[0][1]].data["train_mask"]
    val_mask = hg.edges[hg.canonical_etypes[0][1]].data["val_mask"]
    test_mask = hg.edges[hg.canonical_etypes[0][1]].data["test_mask"]
    train_eidx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_eidx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_eidx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    E = hg.num_edges(hg.canonical_etypes[0][1])
    reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])
    print("First reverse id is:  ", reverse_eids[0])

    g = dgl.to_homogeneous(hg)
    g = g.to("cuda" if args.mode == "gpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    train_eidx.to(device)
    val_eidx.to(device)
    test_eidx.to(device)

    vocab_size = g.num_nodes()

    # create sampler & dataloaders
    sampler = NeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )
    # no neighbor sampling during test
    test_sampler = MultiLayerFullNeighborSampler(1)
    test_sampler = as_edge_prediction_sampler(
        test_sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )

    use_uva = args.mode == "mixed"

    train_dataloader = DataLoader(
        g,
        train_eidx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_dl_workers,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_eidx,
        sampler,
        device=device,
        batch_size=args.batch_size_eval,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_dl_workers,
        use_uva=use_uva,
    )

    test_dataloader = DataLoader(
        g,
        test_eidx,
        test_sampler,
        device=device,
        batch_size=args.batch_size_eval,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_dl_workers,
        use_uva=use_uva,
    )

    model = Model(vocab_size, args.num_hidden, args.num_layers).to(device)

    # model training
    print("Training...")
    train(args, device, g, train_dataloader, val_dataloader, test_dataloader, model)

    # to load pretrained model if you just want to do inference
    #    model.load_state_dict(torch.load(args.model_file))
    #    node_emb = model.emb.weight.data
    #    torch.save(node_emb, args.node_embeddings_file)

    print("Inference to generate node representations...")
    model.eval()
    model.load_state_dict(torch.load(args.model_file))
    node_emb = model.inference(g, device, args.batch_size_eval)
    print("Node embeddings shape: ", node_emb.shape)
    torch.save(node_emb, args.node_embeddings_file)

    # roc_auc on the different splits after training completion
    # roc_auc_test = evaluate(model, test_dataloader)
    # print("final test roc_auc", roc_auc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph SAGE")
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "gpu", "mixed"],
        help="Training mode. 'cpu' for CPU training, 'gpu' for pure-GPU training, "
             "'mixed' for CPU-GPU mixed training.")
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="Path to CSVDataset")
    parser.add_argument(
        "--model_file",
        default="model_graphsage_2L_64.pt",
        type=str,
        help="output for model /your_path/model_graphsage_2L_64.pt")
    parser.add_argument(
        "--node_embeddings_file",
        default="node_emb.pt",
        type=str,
        help="node emb output: /your_path/node_emb.pt")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)

    parser.add_argument("--num_dl_workers", type=int, default=4)

    args = parser.parse_args()
    print(args)

    main(args)
