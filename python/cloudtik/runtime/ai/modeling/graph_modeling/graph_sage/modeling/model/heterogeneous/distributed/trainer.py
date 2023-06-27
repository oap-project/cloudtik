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
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl

import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.distributed.utils import get_eids_mask, get_eids_from_mask, \
    get_edge_split_indices, save_node_embeddings
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import tensor_dict_flatten, tensor_dict_shape
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.utils import \
    parse_reverse_edges


class Trainer:
    def __init__(self, model, model_eval, args):
        self.model = model
        self.model_eval = model_eval
        self.args = args

    def _get_graph_model(self, model):
        """Return the real graph model"""
        if not self.args.standalone:
            return model.module
        else:
            return model

    def train(self, graph):
        args = self.args
        relations = args.relations

        # load the partitioned graph (from homogeneous)
        print("Load partitioned graph")
        dgl.distributed.initialize(args.ip_config)
        if not args.standalone:
            torch.distributed.init_process_group(backend="gloo")
        g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)

        print("Read train/test/val from:", relations)
        print("Will shuffle edges according to edge mapping from global")
        emap = torch.load(os.path.join(os.path.dirname(args.part_config), "emap.pt"))
        # It's a dict of edge mapping
        print("Edge mapping shape:", tensor_dict_shape(emap))

        # The reverse types must guarantee a reverse edge has the same id
        reverse_etypes = None
        if args.exclude_reverse_edges and args.reverse_edges:
            reverse_etypes = parse_reverse_edges(args.reverse_edges)
            print("Reverse edges be excluded during the training:", reverse_etypes)

        train_eids_mask = get_eids_mask(
            graph, relations, "train_mask", reverse_etypes)
        val_eids = get_eids_from_mask(
            graph, relations, "val_mask", emap, reverse_etypes)
        test_eids = get_eids_from_mask(
            graph, relations, "test_mask", emap, reverse_etypes)

        # The mask ids here are the original ids
        # the result is the shuffled ids by the partition book
        train_eids = get_edge_split_indices(
            g, relations,
            train_eids_mask,
        )

        if args.num_gpus == -1:
            device = torch.device("cpu")
        else:
            dev_id = g.rank() % args.num_gpus
            device = torch.device("cuda:" + str(dev_id))

        train_dataloader, val_dataloader, test_dataloader = self._create_data_loaders(
            g, train_eids, val_eids, test_eids, reverse_etypes
        )

        self._train(
            args, device, g,
            train_dataloader, val_dataloader, test_dataloader)
        print("Train ends")

    def _create_data_loaders(
            self, g, train_eids, val_eids, test_eids, reverse_etypes):
        args = self.args

        # Create sampler
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.num_negs)
        sampler = dgl.dataloading.NeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(",")]
        )
        test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        # Create dataloader
        # "reverse_id" exclude not only edges in minibatch but their reverse edges according to reverse_eids mapping
        # reverse_eids - The i-th element indicates the ID of the i-th edge’s reverse edge.
        exclude = "reverse_types" if reverse_etypes is not None else None
        train_dataloader = dgl.dataloading.DistEdgeDataLoader(
            g,
            train_eids,
            sampler,
            negative_sampler=neg_sampler,
            exclude=exclude,
            reverse_etypes=reverse_etypes,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dataloader = dgl.dataloading.DistEdgeDataLoader(
            g,
            val_eids,
            sampler,
            negative_sampler=neg_sampler,
            exclude=exclude,
            reverse_etypes=reverse_etypes,
            batch_size=args.batch_size_eval,
            shuffle=True,
            drop_last=False,
        )

        test_dataloader = dgl.dataloading.DistEdgeDataLoader(
            g,
            test_eids,
            test_sampler,
            negative_sampler=neg_sampler,
            exclude=exclude,
            reverse_etypes=reverse_etypes,
            batch_size=args.batch_size_eval,
            shuffle=False,
            drop_last=False,
        )

        return train_dataloader, val_dataloader, test_dataloader

    def _train(
            self, args, device, g,
            train_dataloader, val_dataloader, test_dataloader):
        model = self.model.to(device)

        if not args.standalone:
            if args.num_gpus == -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                )
            else:
                dev_id = g.rank() % args.num_gpus
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[dev_id],
                    output_device=dev_id,
                )
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        graph_model = self._get_graph_model(model)

        # Training loop
        epoch = 0
        best_model_path = args.model_file
        best_rocauc = 0
        for epoch in range(args.num_epochs):
            num_seeds = 0
            num_inputs = 0
            step_time = []
            sample_t = []
            feat_copy_t = []
            forward_t = []
            backward_t = []
            update_t = []
            iter_tput = []
            total_loss = 0
            start = time.time()
            with model.join():
                # Loop over the dataloader to sample the computation dependency
                # graph as a list of blocks.
                for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(
                    train_dataloader
                ):
                    tic_step = time.time()
                    sample_t.append(tic_step - start)

                    copy_t = time.time()
                    pos_graph = pos_graph.to(device)
                    neg_graph = neg_graph.to(device)
                    blocks = [block.to(device) for block in blocks]
                    feat_copy_t.append(copy_t - tic_step)
                    copy_time = time.time()

                    # Compute loss and prediction
                    x = graph_model.get_inputs(input_nodes, blocks)
                    pos_score, neg_score = model(pos_graph, neg_graph, blocks, x)
                    pos_score = tensor_dict_flatten(pos_score)
                    neg_score = tensor_dict_flatten(neg_score)
                    score = torch.cat([pos_score, neg_score])
                    pos_label = torch.ones_like(pos_score)
                    neg_label = torch.zeros_like(neg_score)
                    labels = torch.cat([pos_label, neg_label])
                    loss = F.binary_cross_entropy_with_logits(score, labels)
                    forward_end = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    compute_end = time.time()
                    forward_t.append(forward_end - copy_time)
                    backward_t.append(compute_end - forward_end)

                    # Aggregate gradients in multiple nodes.
                    optimizer.step()
                    update_t.append(time.time() - compute_end)
                    total_loss += loss.item()
                    pos_edges = pos_graph.num_edges()

                    step_t = time.time() - start
                    step_time.append(step_t)
                    iter_tput.append(pos_edges / step_t)
                    num_seeds += pos_edges
                    if step % args.log_every == 0:
                        print(
                            "[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed "
                            "(samples/sec) {:.4f} | time {:.3f}s | sample {:.3f} | "
                            "copy {:.3f} | forward {:.3f} | backward {:.3f} | "
                            "update {:.3f}".format(
                                g.rank(),
                                epoch,
                                step,
                                loss.item(),
                                np.mean(iter_tput[3:]),
                                np.sum(step_time[-args.log_every :]),
                                np.sum(sample_t[-args.log_every :]),
                                np.sum(feat_copy_t[-args.log_every :]),
                                np.sum(forward_t[-args.log_every :]),
                                np.sum(backward_t[-args.log_every :]),
                                np.sum(update_t[-args.log_every :]),
                            )
                        )
                    start = time.time()

            print(
                "[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, "
                "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
                "#inputs: {}".format(
                    g.rank(),
                    np.sum(step_time),
                    np.sum(sample_t),
                    np.sum(feat_copy_t),
                    np.sum(forward_t),
                    np.sum(backward_t),
                    np.sum(update_t),
                    num_seeds,
                    num_inputs,
                )
            )
            # NOTE: since evaluation is currently done on single rank, after the first epoch,
            # the epoch time reported by ranks other than zero include the evaluation time
            # while they are waiting for epoch to complete.
            if (g.rank() == 0 and epoch % args.eval_every == 0 and epoch != 0) or (
                g.rank() == 0 and epoch == args.num_epochs - 1
            ):
                # load weights into single rank model
                model_eval = self.model_eval.to(
                    device
                )
                model_eval.load_state_dict(graph_model.state_dict())
                # calculate test score on full test set
                with torch.no_grad():
                    rocauc, ap_score = self.evaluate(model_eval, test_dataloader)
                print("Epoch {:05d} | roc_auc {:.4f}".format(epoch, rocauc))
                # update best model if needed
                if best_rocauc < rocauc:
                    print("updating best model")
                    best_rocauc = rocauc
                    torch.save(graph_model.state_dict(), best_model_path)

            # print average epoch loss  per rank
            print(
                "[{}] Epoch {:05d} | Loss {:.4f}".format(
                    g.rank(), epoch, total_loss / (step + 1)
                )
            )
            epoch += 1

        # sync the status for all ranks
        if not args.standalone:
            torch.distributed.barrier()

        # train is complete, save node embeddings of the best model
        # sync for eval and test
        best_model_path = args.model_file
        graph_model = self._get_graph_model(model)
        graph_model.load_state_dict(torch.load(best_model_path))
        if not args.standalone:
            # save node embeddings into file
            torch.nn.Module.eval(model)  # model.eval()
            with torch.no_grad():
                x = graph_model.get_inference_inputs(g)
                node_emb = graph_model.inference(g, x, args.batch_size_eval, device)
            if g.rank() == 0:
                save_node_embeddings(node_emb, args.node_embeddings_file)
            g._client.barrier()
        else:
            torch.nn.Module.eval(model)  # model.eval()
            # save node embeddings into file
            with torch.no_grad():
                x = graph_model.get_inference_inputs(g)
                node_emb = graph_model.inference(g, x, args.batch_size_eval, device)
                save_node_embeddings(node_emb, args.node_embeddings_file)

        if not args.standalone:
            torch.distributed.barrier()
            # torch.distributed.monitored_barrier(timeout=timedelta(minutes=60))

    def evaluate(self, model, test_dataloader):
        # evaluate the embeddings on test set
        torch.nn.Module.eval(model)  # model.eval()
        score_all = torch.tensor(())
        labels_all = torch.tensor(())
        with torch.no_grad():
            for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
                tqdm.tqdm(test_dataloader)
            ):
                x = model.get_inputs(input_nodes, blocks)
                pos_score, neg_score = model(
                    pair_graph, neg_pair_graph, blocks, x
                )
                # Heterogeneous: handle the result as dict of score of each edge types
                pos_score = tensor_dict_flatten(pos_score)
                neg_score = tensor_dict_flatten(neg_score)
                score = torch.cat([pos_score, neg_score])
                pos_label = torch.ones_like(pos_score)
                neg_label = torch.zeros_like(neg_score)
                labels = torch.cat([pos_label, neg_label])
                score_all = torch.cat([score_all, score])
                labels_all = torch.cat([labels_all, labels])
            rocauc = roc_auc_score(labels_all, score_all)
            ap_score = average_precision_score(labels_all, score_all)
            return rocauc, ap_score
