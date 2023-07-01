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

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score

from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    MultiLayerFullNeighborSampler,
)

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    homogeneous.utils import get_eids_from_mask, _create_edge_prediction_sampler, get_reverse_eids
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.utils import \
    parse_reverse_edges, get_common_node_features, get_common_edge_features


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.graph = None
        self.model = model

    def train(self, graph, device):
        args = self.args

        print("Read train/test/val from:", graph.etypes)

        reverse_etypes = None
        if args.exclude_reverse_edges and args.reverse_edges:
            reverse_etypes = parse_reverse_edges(args.reverse_edges)
            print("Reverse edges be excluded during the training:", reverse_etypes)

        train_eids = get_eids_from_mask(graph, "train_mask", reverse_etypes)
        val_eids = get_eids_from_mask(graph, "val_mask", reverse_etypes)
        test_eids = get_eids_from_mask(graph, "test_mask", reverse_etypes)

        reverse_eids = None
        if args.exclude_reverse_edges and reverse_etypes:
            # The i-th element indicates the ID of the i-th edge’s reverse edge.
            reverse_eids = get_reverse_eids(graph, reverse_etypes)

        ndata = get_common_node_features(graph)
        edata = get_common_edge_features(graph)
        g = dgl.to_homogeneous(graph, ndata=ndata, edata=edata)
        g = g.to("cuda" if args.mode == "gpu" else "cpu")
        self.graph = g

        train_eids = train_eids.to(device)
        val_eids = val_eids.to(device)
        test_eids = test_eids.to(device)

        train_dataloader, val_dataloader, test_dataloader = self._create_data_loaders(
            g, device, train_eids, val_eids, test_eids, reverse_eids
        )

        # model training
        print("Training...")
        self._train(
            args, device, g,
            train_dataloader, val_dataloader, test_dataloader)

        return self.model

    def _create_data_loaders(
            self, g, device, train_eids, val_eids, test_eids, reverse_eids):
        args = self.args

        # create sampler & dataloaders
        sampler = NeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
        sampler = _create_edge_prediction_sampler(
            sampler, reverse_eids=reverse_eids
        )
        # no neighbor sampling during test
        test_sampler = MultiLayerFullNeighborSampler(1)
        test_sampler = _create_edge_prediction_sampler(
            test_sampler, reverse_eids=reverse_eids
        )

        use_uva = args.mode == "mixed"

        train_dataloader = DataLoader(
            g,
            train_eids,
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
            val_eids,
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
            test_eids,
            test_sampler,
            device=device,
            batch_size=args.batch_size_eval,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_dl_workers,
            use_uva=use_uva,
        )

        return train_dataloader, val_dataloader, test_dataloader

    def evaluate(self, model, g, test_dataloader):
        model.eval()
        score_all = torch.tensor(())
        labels_all = torch.tensor(())
        with test_dataloader.enable_cpu_affinity():
            with torch.no_grad():
                for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                    tqdm.tqdm(test_dataloader)
                ):
                    # x = blocks[0].srcdata[dgl.NID]  # this is the same as input_nodes for homogeneous only
                    x = model.get_inputs(g, input_nodes, blocks)
                    pos_score, neg_score = model(
                        pair_graph, neg_pair_graph, blocks, x)
                    score = torch.cat([pos_score, neg_score])
                    pos_label = torch.ones_like(pos_score)
                    neg_label = torch.zeros_like(neg_score)
                    labels = torch.cat([pos_label, neg_label])
                    score_all = torch.cat([score_all, score])
                    labels_all = torch.cat([labels_all, labels])
                rocauc = roc_auc_score(labels_all, score_all)
        return rocauc

    def _train(
            self, args, device, g, train_dataloader, val_dataloader, test_dataloader):
        model = self.model.to(device)
        best_rocauc = 0
        best_model_path = args.model_file
        print("learning rate:", args.lr)
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
                    # x = blocks[0].srcdata[dgl.NID]  # this is the same as input_nodes for homogeneous only
                    x = model.get_inputs(g, input_nodes, blocks)
                    pos_score, neg_score = model(
                        pair_graph, neg_pair_graph, blocks, x
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
                        # x = blocks[0].srcdata[dgl.NID]  # this is the same as input_nodes for homogeneous only
                        x = model.get_inputs(g, input_nodes, blocks)
                        pos_score, neg_score = model(
                            pair_graph, neg_pair_graph, blocks, x
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
                    rocauc = self.evaluate(model, g, test_dataloader)
                    # update best model if needed
                    if best_rocauc < rocauc:
                        print("Updating best model")
                        best_rocauc = rocauc
                        torch.save(model.state_dict(), best_model_path)
                    print("Epoch {:05d} | Test roc_auc_score {:.4f}".format(epoch, rocauc))
