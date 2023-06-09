# Modifications Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import time

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import dgl

import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.distributed.model \
    import DistributedGraphSAGEModel


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

    # encoder/decoder with DistributedGraphSAGE
    model = DistributedGraphSAGEModel(g.num_nodes(), args.num_hidden, args.num_layers).to(device)
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
    model_noDDP = DistributedGraphSAGEModel(g.num_nodes(), args.num_hidden, args.num_layers).to(device)

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
            model_noDDP = DistributedGraphSAGEModel(g.num_nodes(), args.num_hidden, args.num_layers).to(
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
