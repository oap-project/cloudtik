# Modifications Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import argparse
import time
import os
import random
from contextlib import contextmanager

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn import SAGEConv

import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score


class DistSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type
    ):
        super(DistSAGE, self).__init__()
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


#################
# Define model


class Model(nn.Module):
    def __init__(self, vocab_size, hid_size, num_layers):
        super().__init__()

        self.hid_size = hid_size
        # node embedding
        self.emb = th.nn.Embedding(vocab_size, hid_size)

        # encoder is a 1-layer GraphSAGE model
        self.encoder = DistSAGE(
            hid_size, hid_size, hid_size, num_layers, F.relu, "mean"
        )
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        h = self.encoder(blocks, h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.hid_size),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.encoder.layers):
            if i == len(self.encoder.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.hid_size),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.encoder.layers) - 1:
                    h = self.encoder.activation(h)
                #                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


######


def run(args, device, data):
    # Unpack data
    (
        train_eids,
        val_eids,
        test_eids,
        reverse_eids,
        g,
    ) = data
    # Create sampler

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.num_negs)
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    test_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # Create dataloader
    # "reverse_id" exclude not only edges in minibatch but their reverse edges according to reverse_eids mapping
    # reverse_eids - The i-th element indicates the ID of the i-th edge’s reverse edge.
    exclude = "reverse_id" if args.remove_edge else None
    # reverse_eids = th.arange(g.num_edges()) if args.remove_edge else None
    train_dataloader = dgl.dataloading.DistEdgeDataLoader(
        g,
        train_eids,
        sampler,
        negative_sampler=neg_sampler,
        exclude=exclude,
        reverse_eids=reverse_eids,
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
        reverse_eids=reverse_eids,
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
        reverse_eids=reverse_eids,
        batch_size=args.batch_size_eval,
        shuffle=False,
        drop_last=False,
    )

    # encoder/decoder with DistSAGE
    model = Model(g.num_nodes(), args.num_hidden, args.num_layers).to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(
                model,
            )
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dev_id],
                output_device=dev_id,
            )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if not args.standalone:
        node_emb = model.module.emb.weight.data
        # print(node_emb.shape)
    else:
        node_emb = model.emb.weight.data
        # print(node_emb.shape)

    # define copy of model not DDP for single node evaluation
    model_noDDP = Model(g.num_nodes(), args.num_hidden, args.num_layers).to(device)

    # # Training loop
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
                pos_score, neg_score = model(pos_graph, neg_graph, blocks, input_nodes)
                score = th.cat([pos_score, neg_score])
                pos_label = th.ones_like(pos_score)
                neg_label = th.zeros_like(neg_score)
                labels = th.cat([pos_label, neg_label])
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
            model_noDDP = Model(g.num_nodes(), args.num_hidden, args.num_layers).to(
                device
            )
            model_noDDP.load_state_dict(model.module.state_dict())
            # calculate test score on full test set
            with th.no_grad():
                rocauc, ap_score = evaluate(model_noDDP, test_dataloader)
            print("Epoch {:05d} | roc_auc {:.4f}".format(epoch, rocauc))
            # update best model if needed
            if best_rocauc < rocauc:
                print("updating best model")
                best_rocauc = rocauc
                th.save(model.module.state_dict(), best_model_path)

        # print average epoch loss  per rank
        print(
            "[{}] Epoch {:05d} | Loss {:.4f}".format(
                g.rank(), epoch, total_loss / (step + 1)
            )
        )
        epoch += 1

    # sync the status for all ranks
    if not args.standalone:
        th.distributed.barrier()

    # train is complete, save node embeddings of the best model
    # sync for eval and test
    best_model_path = args.model_file
    model.module.load_state_dict(th.load(best_model_path))
    if not args.standalone:
        # save node embeddings into file
        th.nn.Module.eval(model)  # model.eval()
        with th.no_grad():
            x = model.module.emb.weight.data
            print(x.shape)
            node_emb = model.module.inference(g, x, args.batch_size_eval, device)
        if g.rank() == 0:
            th.save(node_emb[0 : g.num_nodes()], args.node_embeddings_file)
            print("node emb shape: ", node_emb.shape)
        g._client.barrier()
    else:
        th.nn.Module.eval(model)  # model.eval()
        # save node embeddings into file
        with th.no_grad():
            x = model.emb.weight.data
            # print(x.shape)
            node_emb = model.module.inference(g, x, args.batch_size_eval, device)

            th.save(node_emb, args.node_embeddings_file)

    if not args.standalone:
        th.distributed.barrier()
        # th.distributed.monitored_barrier(timeout=timedelta(minutes=60))


def evaluate(model, test_dataloader):
    # evaluate the embeddings on test set
    th.nn.Module.eval(model)  # model.eval()
    score_all = th.tensor(())
    labels_all = th.tensor(())
    with th.no_grad():
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            tqdm.tqdm(test_dataloader)
        ):
            pos_score, neg_score = model(
                pair_graph, neg_pair_graph, blocks, input_nodes
            )
            score = th.cat([pos_score, neg_score])
            pos_label = th.ones_like(pos_score)
            neg_label = th.zeros_like(neg_score)
            labels = th.cat([pos_label, neg_label])
            score_all = th.cat([score_all, score])
            labels_all = th.cat([labels_all, labels])
        rocauc = roc_auc_score(labels_all, score_all)
        ap_score = average_precision_score(labels_all, score_all)
        return rocauc, ap_score


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
