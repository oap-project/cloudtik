# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn

# For distributed run
import extend_distributed as ext_dist


import sklearn.metrics
import mlperf_logger
from torch.profiler import schedule
from torch.profiler import profile, record_function, ProfilerActivity
# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

exc = getattr(builtins, "IOError", "FileNotFoundError")
first_iteration = True
class AllReduceRequest():
    def __init__(self):
        self.requset = None
        self.tensor = None
        self.grad_name = None
        self.wait = False #call wait at the end of the function backback

class AlltoallOutputs():
    def __init__(self):
        self.data = None
        self.tail_data = None

    def get_data_base_on_input(self, input, shape):
        if self.data is None:
            self.data = input.new_empty(shape)
            return self.data
        if self.data.shape == torch.Size(shape):
            return self.data
        else:
            if self.tail_data is None:
                self.tail_data = input.new_empty(shape)
            return self.tail_data

class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        if isinstance(optimizer, tuple):
            for opt in optimizer:
                super(LRPolicyScheduler, self).__init__(opt)
        else:
            super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):

    def create_mlp(self, ln, sigmoid_layer, tensor_name):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

            if ext_dist.my_size == 1 or ext_dist.dist.get_rank() == 0:
                mlperf_logger.log_event(key=mlperf_logger.constants.WEIGHTS_INITIALIZATION,  metadata={'tensor': tensor_name + str(i+1)})
        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, local_ln_emb_sparse=None, ln_emb_dense=None):
        emb_l = nn.ModuleList()
        # save the numpy random state
        #np_rand_state = np.random.get_state()
        emb_dense = nn.ModuleList()
        emb_sparse = nn.ModuleList()
        embs = range(len(ln))
        if local_ln_emb_sparse or ln_emb_dense:
            embs = local_ln_emb_sparse + ln_emb_dense
        for i in embs:
            # Use per table random seed for Embedding initialization
            #np.random.seed(self.l_emb_seeds[i])
            n = ln[i]
            print("Create Embedding: {}".format(n), flush=True)
            # approach 1
            if n >= self.sparse_dense_boundary:
                # For sparse embs, split the table across ranks along sparse dimension
                if (ext_dist.my_size > 1) and (ext_dist.my_size > len(self.ln_emb_sparse)):
                    new_m = m // (ext_dist.my_size // len(self.ln_emb_sparse))
                else:
                    new_m = m

                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, new_m)
                ).astype(np.float32)

                EE = nn.EmbeddingBag(n, new_m, mode="sum", sparse=True, _weight=torch.tensor(W, requires_grad=True))
                emb_sparse.append(EE)
            else:
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)

                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False, _weight=torch.tensor(W, requires_grad=True))
                emb_dense.append(EE)
        if not self.hybrid_gradient_emb:
            emb_l.append(EE)

        if ext_dist.my_size == 1 or ext_dist.dist.get_rank() == 0:
            mlperf_logger.log_event(key=mlperf_logger.constants.WEIGHTS_INITIALIZATION,  metadata={'tensor': 'embeddings'})

        return emb_l, emb_dense, emb_sparse

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        bf16=False,
        use_ipex= True,
        sparse_dense_boundary = 2048,
        hybrid_gradient_emb = False,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            #self.parallel_model_batch_size = -1
            #self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.bf16 = bf16
            self.use_ipex = use_ipex
            self.sparse_dense_boundary = sparse_dense_boundary
            self.hybrid_gradient_emb = hybrid_gradient_emb
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # generate np seeds for Emb table initialization
            self.l_emb_seeds = np.random.randint(low=0, high=100000, size=len(ln_emb))
            print("#############seed self.l_emb_seeds:{}".format(self.l_emb_seeds))
            n_emb = len(ln_emb)
            self.ln_emb_dense = [i for i in range(n_emb) if ln_emb[i] < self.sparse_dense_boundary]
            self.ln_emb_sparse = [i for i in range(n_emb) if ln_emb[i] >= self.sparse_dense_boundary]
            #If running distributed, get local slice of embedding tables
            self.rank = -1
            if ext_dist.my_size > 1:
                self.rank = ext_dist.dist.get_rank()
                n_emb_sparse = len(self.ln_emb_sparse)
                self.n_local_emb_sparse, self.n_sparse_emb_per_rank = ext_dist.get_split_lengths(n_emb_sparse, split=True)
                self.local_ln_emb_sparse_slice = ext_dist.get_my_slice(n_emb_sparse)
                self.local_ln_emb_sparse = self.ln_emb_sparse[self.local_ln_emb_sparse_slice]

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot, "bottom_mlp_dense")
            self.top_l = self.create_mlp(ln_top, sigmoid_top, "top_mlp_dense")
            self.emb_dense_flat_grads = None
            self.emb_dense_ar_req = AllReduceRequest()
            self.top_mlp_flat_grads = None
            self.top_mlp_ar_req = AllReduceRequest()
            self.bot_mlp_flat_grads = None
            self.bot_mlp_ar_req = AllReduceRequest()
            self.all2all_train_outputs = AlltoallOutputs()
            self.all2all_validation_outputs0 = AlltoallOutputs()
            self.all2all_validation_outputs1 = AlltoallOutputs()
            self.output_ind = AlltoallOutputs()
            self.validation_output_ind0 = AlltoallOutputs()
            self.validation_output_ind1 = AlltoallOutputs()
            self.start_train_iteration = True
            # create operators
            if ndevices <= 1:
                if ext_dist.my_size > 1:
                    self.emb_l, self.emb_dense, self.emb_sparse = self.create_emb(m_spa, ln_emb, self.local_ln_emb_sparse, self.ln_emb_dense)
                elif self.hybrid_gradient_emb:
                    _, self.emb_dense, self.emb_sparse = self.create_emb(m_spa, ln_emb)
                else:
                    self.emb_l, _, _ = self.create_emb(m_spa, ln_emb)
    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb_merged(self, batch_size, lS_i, emb_l):
        emb_l.linearize_merged_indices_1D(lS_i, batch_size)
        merged_input = (lS_i, emb_l.merged_offsets, emb_l.merged_indices_with_row_offsets)
        if ext_dist.my_size > 1:
            if isinstance(emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD) and self.emb_dense_flat_grads is  not None:#overlap the allreuce of dense emb  with backward of sparse emb
                return emb_l(merged_input, self.need_linearize_indices_and_offsets, self.emb_dense_ar_req)
            elif self.bot_mlp_flat_grads is not None:#issue the allreduce of bot mlp in the begin of backward of emb dense
                return emb_l(merged_input, self.need_linearize_indices_and_offsets, self.bot_mlp_ar_req)
            else:
                return emb_l(merged_input, self.need_linearize_indices_and_offsets)
        else:
            return emb_l(merged_input, self.need_linearize_indices_and_offsets)

    def apply_emb(self, batch_size, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        if args.use_ipex and isinstance(emb_l, ipex.nn.modules.MergedEmbeddingBag):
            return self.apply_emb_merged(batch_size, lS_i, emb_l)
        ly = []
        batch_size = lS_i[0].numel()
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = torch.arange(batch_size).reshape(1, -1)[0]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            if args.ipex_interaction and args.use_ipex:
                T = [x] + list(ly)
                if ext_dist.my_size > 1 and torch.is_grad_enabled() and self.top_mlp_flat_grads is not None:
                    R = ipex.nn.functional.interaction(self.top_mlp_ar_req, *T)
                else:
                    R = ipex.nn.functional.interaction(None, *T)
            else:
                # concatenate dense and sparse features
                (batch_size, d) = x.shape
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
                # perform a dot product
                Z = torch.bmm(T, torch.transpose(T, 1, 2))
                # append dense feature with the interactions (into a row vector)
                # approach 1: all
                # Zflat = Z.view((batch_size, -1))
                # approach 2: unique
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_i_sparse, lS_i_dense, is_train=True):
        batch_size = dense_x.size()[0]
        if ext_dist.my_size > 1:
            return self.distributed_forward(batch_size, dense_x, lS_i_sparse, lS_i_dense, is_train)
        elif self.ndevices <= 1:
            return self.sequential_forward(batch_size, dense_x, lS_i_sparse, lS_i_dense)
        else:
            assert("Not supported !")
            #return self.parallel_forward(dense_x, lS_i)

    def sequential_forward(self, batch_size, dense_x, lS_i_sparse, lS_i_dense):
        # process dense features (using bottom mlp), resulting in a row vector
        with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
            x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())
        ly = None
        # process sparse features(using embeddings), resulting in a list of row vectors
        if self.hybrid_gradient_emb:
            #lS_o_dense = [lS_o[i]  for i in self.ln_emb_dense]
            #lS_o_sparse = [lS_o[i] for i in self.ln_emb_sparse]  # partition sparse table in one group
            #lS_i_dense = [lS_i[i]  for i in self.ln_emb_dense]
            #lS_i_sparse = [lS_i[i]  for i in self.ln_emb_sparse]
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
                ly_dense = self.apply_emb(batch_size, lS_i_dense, self.emb_dense)
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
                ly_sparse = self.apply_emb(batch_size, lS_i_sparse, self.emb_sparse)
            ly = ly_dense + ly_sparse
        else:
            ly = self.apply_emb(batch_size, lS_i_sparse, self.emb_l)

        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        with torch.autograd.profiler.record_function('Prof_interaction_forward'):
            z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
            p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def distributed_forward(self, batch_size, dense_x, lS_i_sparse, lS_i_dense, is_train):
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        #if batch_size < ext_dist.my_size:
        #    sys.exit("ERROR: batch_size (%d) must be larger than number of ranks (%d)" % (batch_size, ext_dist.my_size))

        #lS_o_sparse = [lS_o[i] for i in self.ln_emb_sparse]  # partition sparse table in one group
        if args.use_hybridparallel_dataset:
            #dense_embs_num = len(self.ln_emb_dense)
            #lS_i_dense = [lS_i[i] for i in range(dense_embs_num)]
            g_i_sparse = []
            for i in range(len(self.local_ln_emb_sparse)):
                global_sparse_index = lS_i_sparse[i * ext_dist.my_size : (i + 1) * ext_dist.my_size]
                global_sparse_index = global_sparse_index.reshape(-1)
                g_i_sparse.append(global_sparse_index)
                #offset = torch.arange(batch_size * ext_dist.my_size).to(device)
                #g_o_sparse = [offset[self.n_local_emb_sparse[0]]]
            #if (len(self.local_ln_emb_sparse) != len(g_o_sparse)) or (len(self.local_ln_emb_sparse) != len(g_i_sparse)):
            #   sys.exit("ERROR 0 : corrupted model input detected in distributed_forward call")
            if (len(self.local_ln_emb_sparse) != len(g_i_sparse)):
               sys.exit("ERROR 0 : corrupted model input detected in distributed_forward call")
            # sparse embeddings
            ly_sparse = self.apply_emb(batch_size, g_i_sparse, self.emb_sparse)
            t_alltoall_emb_begin = time.time()
            if is_train:
                a2a_req = ext_dist.alltoall(self.all2all_train_outputs, ly_sparse, self.n_sparse_emb_per_rank)
            else:
                a2a_req = ext_dist.alltoall(self.all2all_validation_outputs0, ly_sparse, self.n_sparse_emb_per_rank)
            t_alltoall_emb_end = time.time()
            if self.rank == 0:
                print("alltoall_emb_forward_time: {}".format(t_alltoall_emb_end - t_alltoall_emb_begin))
            # dense embeddings
            ly_dense = self.apply_emb(batch_size, lS_i_dense, self.emb_dense)

        else:
            global first_iteration
            if first_iteration or not is_train:
                if ext_dist.my_size > len(self.ln_emb_sparse):
                    num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                    lS_i_sparse = torch.cat([lS_i_sparse for _ in range(num_split_grps) ])
                output = lS_i_sparse.new_empty(lS_i_sparse.size())
                req = ext_dist.dist.all_to_all_single(output, lS_i_sparse, async_op=True)
                req.wait()
                lS_i_sparse = output.reshape(ext_dist.my_size, -1)
                g_i_sparse = [lS_i_sparse[:, i * batch_size:(i + 1) * batch_size].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
                lS_i_sparse = torch.cat(g_i_sparse)
                first_iteration = False
            #    g_i_sparse = torch.cat(g_i_sparse)
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
            # sparse embeddings
                ly_sparse = self.apply_emb(batch_size*ext_dist.my_size, lS_i_sparse, self.emb_sparse)
            #t_alltoall_emb_begin = time.time()
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward'):
                if is_train:
                    a2a_req = ext_dist.alltoall(self.all2all_train_outputs, ly_sparse, self.n_sparse_emb_per_rank)
                else:
                    a2a_req = ext_dist.alltoall(self.all2all_validation_outputs0, ly_sparse, self.n_sparse_emb_per_rank)
            if is_train:
                load_data(data_iter, buffer_num)

            #t_alltoall_emb_end = time.time()
            #if self.rank == 0:
            #    print("alltoall_emb_forward_time: {}ms".format(1000*(t_alltoall_emb_end - t_alltoall_emb_begin)))
            # dense embeddings
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
                if is_train and not self.start_train_iteration and args.ipex_merged_emb and args.hybrid_gradient_emb:
                    optimizers[1].step()#overlap the dense_emb
                    lr_schedulers[1].step()
                if is_train:
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                ly_dense = self.apply_emb(batch_size, lS_i_dense, self.emb_dense)

        # bottom mlp
        with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
            x = self.apply_mlp(dense_x, self.bot_l)
        with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward'):
            ly_sparse = a2a_req.wait()


        # concat emb data for split sparse embs
        ly_sparse_full = []
        if ext_dist.my_size > len(self.ln_emb_sparse):
            # for i in range(len(self.ln_emb_sparse)):
            #     ly_sparse_split = torch.cat([ly_sparse[j] for j in range(i, ext_dist.my_size, 16)], 1)
            #     ly_sparse_full.append(ly_sparse_split)
            for i in range(len(self.ln_emb_sparse)):
                ly_sparse_split = torch.cat([ly_sparse[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                ly_sparse_full.append(ly_sparse_split)
        else:
            ly_sparse_full = list(ly_sparse)

        ly = list(ly_dense) + ly_sparse_full

        a2a_ind_req = None #ovlerlap the a2a_ind_req with interaction/top_mlp
        if is_train:#get global index for sparse embedding
            (X_next, T_next, lS_i_sparse_next, lS_i_dense_next) = data_buffer[0]
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                lS_i_sparse_next = torch.cat([lS_i_sparse_next for _ in range(num_split_grps) ])
            output_ind = self.output_ind.get_data_base_on_input(lS_i_sparse_next, lS_i_sparse_next.size())
            a2a_ind_req = ext_dist.dist.all_to_all_single(output_ind, lS_i_sparse_next, async_op=True)
        # interactions
        with torch.autograd.profiler.record_function('Prof_interaction_forward'):
            z = self.interact_features(x, ly)
        # top mlp
        with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
            p = self.apply_mlp(z, self.top_l)

        if is_train:
            a2a_ind_req.wait()
            lS_i_sparse_next = output_ind.reshape(ext_dist.my_size, -1)
            batch_size_next = X_next.size()[0]
            g_i_sparse = [lS_i_sparse_next[:, i * batch_size_next:(i + 1) * batch_size_next].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            lS_i_sparse_next = torch.cat(g_i_sparse)
            data_buffer[0] = (X_next, T_next, lS_i_sparse_next, lS_i_dense_next)
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(
                p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z = p
        if is_train:
            self.start_train_iteration = False
        return z

    def validation(self, cur_data, next_data):
        if not cur_data[0]:
            return None, None

        dense_x0 = cur_data[0]['dense_x']
        T_test0 = cur_data[0]['T_test']
        lS_i_sparse0 = cur_data[0]['lS_i_sparse']
        lS_i_dense0 = cur_data[0]['lS_i_dense']
        batch_size0 = dense_x0.size()[0]

        if not cur_data[1]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                lS_i_sparse0 = torch.cat([lS_i_sparse0 for _ in range(num_split_grps) ])
            output0 = lS_i_sparse0.new_empty(lS_i_sparse0.size())
            ind_req0 = ext_dist.dist.all_to_all_single(output0, lS_i_sparse0, async_op=True)
            ind_req0.wait()
            lS_i_sparse0 = output0.reshape(ext_dist.my_size, -1)
            g_i_sparse0 = [lS_i_sparse0[:, i * batch_size0:(i + 1) * batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            lS_i_sparse0 = torch.cat(g_i_sparse0)

            #    g_i_sparse = torch.cat(g_i_sparse)
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
            # sparse embeddings
                ly_sparse0 = self.apply_emb(batch_size0*ext_dist.my_size, lS_i_sparse0, self.emb_sparse)

            with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward'):
                a2a_req0 = ext_dist.alltoall(self.all2all_validation_outputs0, ly_sparse0, self.n_sparse_emb_per_rank)

            # dense embeddings
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
                ly_dense0 = self.apply_emb(batch_size0, lS_i_dense0, self.emb_dense)

            # bottom mlp
            # with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
            x0 = self.apply_mlp(dense_x0, self.bot_l)
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward'):
                    ly_sparse0 = a2a_req0.wait()

            # concat emb data for split sparse embs
            ly_sparse0_full = []
            if ext_dist.my_size > len(self.ln_emb_sparse):
                for i in range(len(self.ln_emb_sparse)):
                    ly_sparse0_split = torch.cat([ly_sparse0[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                    ly_sparse0_full.append(ly_sparse0_split)
            else:
                ly_sparse0_full = list(ly_sparse0)

            ly0 = list(ly_dense0) + ly_sparse0_full

            # with torch.autograd.profiler.record_function('Prof_interaction_forward'):
            z0 = self.interact_features(x0, ly0)
            # top mlp
            # with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
            p0 = self.apply_mlp(z0, self.top_l)

            # clamp output if needed
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z0 = torch.clamp(
                    p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
                )
            else:
                z0 = p0
            z0 = z0.float()

            Z_test0 = ext_dist.all_gather(z0, None)
            T_test0 = ext_dist.all_gather(T_test0, None)

            return (Z_test0, T_test0), None

        if cur_data[1]:
            dense_x1 = cur_data[1]['dense_x']
            T_test1 = cur_data[1]['T_test']
            lS_i_sparse1 = cur_data[1]['lS_i_sparse']
            lS_i_dense1 = cur_data[1]['lS_i_dense']
            batch_size1 = dense_x1.size()[0]

        if next_data[0]:
            next_dense_x0 = next_data[0]['dense_x']
            # next_T_test0 = next_data[0]['T_test']
            next_lS_i_sparse0 = next_data[0]['lS_i_sparse']
            # next_lS_i_dense0 = next_data[0]['lS_i_dense']
            next_batch_size0 = next_dense_x0.size()[0]

        if next_data[1]:
            next_dense_x1 = next_data[1]['dense_x']
            next_T_test1 = next_data[1]['T_test']
            next_lS_i_sparse1 = next_data[1]['lS_i_sparse']
            next_lS_i_dense1 = next_data[1]['lS_i_dense']
            next_batch_size1 = next_dense_x1.size()[0]

        if 'lS_i_dense_res' not in cur_data[0]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                lS_i_sparse0 = torch.cat([lS_i_sparse0 for _ in range(num_split_grps) ])
            output0 = lS_i_sparse0.new_empty(lS_i_sparse0.size())
            ind_req0 = ext_dist.dist.all_to_all_single(output0, lS_i_sparse0, async_op=True)
            ind_req0.wait()
            lS_i_sparse0 = output0.reshape(ext_dist.my_size, -1)
            g_i_sparse0 = [lS_i_sparse0[:, i * batch_size0:(i + 1) * batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            lS_i_sparse0 = torch.cat(g_i_sparse0)
        else:
            lS_i_sparse0 = cur_data[0]['lS_i_dense_res']

        if cur_data[1] and 'lS_i_dense_res' not in cur_data[1]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                lS_i_sparse1 = torch.cat([lS_i_sparse1 for _ in range(num_split_grps) ])
            output1 = lS_i_sparse1.new_empty(lS_i_sparse1.size())
            ind_req1 = ext_dist.dist.all_to_all_single(output1, lS_i_sparse1, async_op=True)

        if 'ly_sparse' not in cur_data[0]:
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward'):
            # sparse embeddings
                ly_sparse0 = self.apply_emb(batch_size0*ext_dist.my_size, lS_i_sparse0, self.emb_sparse)
        else:
            ly_sparse0 = cur_data[0]['ly_sparse']

        if cur_data[1] and 'lS_i_dense_res' not in cur_data[1]:
            ind_req1.wait()
            lS_i_sparse1 = output1.reshape(ext_dist.my_size, -1)
            g_i_sparse1 = [lS_i_sparse1[:, i * batch_size1:(i + 1) * batch_size1].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            lS_i_sparse1 = torch.cat(g_i_sparse1)
        else:
            lS_i_sparse1 = cur_data[1]['lS_i_dense_res']

        with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward'):
            a2a_req0 = ext_dist.alltoall(self.all2all_validation_outputs0, ly_sparse0, self.n_sparse_emb_per_rank)

        # dense embeddings
        with torch.autograd.profiler.record_function('Prof_dense_emb_forward'):
            ly_dense0 = self.apply_emb(batch_size0, lS_i_dense0, self.emb_dense)

        # bottom mlp
        # with torch.autograd.profiler.record_function('Prof_bot_mlp_forward'):
        x0 = self.apply_mlp(dense_x0, self.bot_l)

        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_sparse_emb_forward1'):
                ly_sparse1 = self.apply_emb(batch_size1*ext_dist.my_size, lS_i_sparse1, self.emb_sparse)

        with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward'):
                ly_sparse0 = a2a_req0.wait()

        T_test_req0, T_test0 = ext_dist.all_gather_validation(T_test0, None)

        # concat emb data for split sparse embs
        ly_sparse0_full = []
        if ext_dist.my_size > len(self.ln_emb_sparse):
            for i in range(len(self.ln_emb_sparse)):
                ly_sparse0_split = torch.cat([ly_sparse0[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                ly_sparse0_full.append(ly_sparse0_split)
        else:
            ly_sparse0_full = list(ly_sparse0)

        ly0 = list(ly_dense0) + ly_sparse0_full

        # with torch.autograd.profiler.record_function('Prof_interaction_forward'):
        z0 = self.interact_features(x0, ly0)
        with torch.autograd.profiler.record_function('Prof_all_gather_t0'):
            T_test_req0.wait()

        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_forward1'):
                a2a_req1 = ext_dist.alltoall(self.all2all_validation_outputs1, ly_sparse1, self.n_sparse_emb_per_rank)

        if next_data[0]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                next_lS_i_sparse0 = torch.cat([next_lS_i_sparse0 for _ in range(num_split_grps) ])
            next_output0 = self.validation_output_ind0.get_data_base_on_input(next_lS_i_sparse0, next_lS_i_sparse0.size())
            next_ind_req0 = ext_dist.dist.all_to_all_single(next_output0, next_lS_i_sparse0, async_op=True)

        # top mlp
        # with torch.autograd.profiler.record_function('Prof_top_mlp_forward'):
        p0 = self.apply_mlp(z0, self.top_l)

        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_alltoall_emb_wait_forward1'):
                    ly_sparse1 = a2a_req1.wait()

        if next_data[0]:
            next_ind_req0.wait()
            next_lS_i_sparse0 = next_output0.reshape(ext_dist.my_size, -1)
            next_g_i_sparse0 = [next_lS_i_sparse0[:, i * next_batch_size0:(i + 1) * next_batch_size0].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            next_lS_i_sparse0 = torch.cat(next_g_i_sparse0)
            next_data[0]['lS_i_dense_res'] = next_lS_i_sparse0

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0
        z0 = z0.float()

        Z_test_req0, Z_test0 = ext_dist.all_gather_validation(z0, None)
        if cur_data[1]:
            with torch.autograd.profiler.record_function('Prof_dense_emb_forward1'):
                ly_dense1 = self.apply_emb(batch_size1, lS_i_dense1, self.emb_dense)
            # bottom mlp
            # with torch.autograd.profiler.record_function('Prof_bot_mlp_forward1'):
            x1 = self.apply_mlp(dense_x1, self.bot_l)

        with torch.autograd.profiler.record_function('Prof_all_gather_z0'):
            Z_test_req0.wait()

        if cur_data[1]:
            T_test_req1, T_test1 = ext_dist.all_gather_validation(T_test1, None)

            # concat emb data for split sparse embs
            ly_sparse1_full = []
            if ext_dist.my_size > len(self.ln_emb_sparse):
                for i in range(len(self.ln_emb_sparse)):
                    ly_sparse1_split = torch.cat([ly_sparse1[j] for j in range(i, ext_dist.my_size, len(self.ln_emb_sparse))], 1)
                    ly_sparse1_full.append(ly_sparse1_split)
            else:
                ly_sparse1_full = list(ly_sparse1)

            ly1 = list(ly_dense1) + ly_sparse1_full

            # with torch.autograd.profiler.record_function('Prof_interaction_forward1'):
            z1 = self.interact_features(x1, ly1)

        if next_data[1]:
            if ext_dist.my_size > len(self.ln_emb_sparse):
                num_split_grps = ext_dist.my_size // len(self.ln_emb_sparse)
                next_lS_i_sparse1 = torch.cat([next_lS_i_sparse1 for _ in range(num_split_grps) ])
            next_output1 = self.validation_output_ind1.get_data_base_on_input(next_lS_i_sparse1, next_lS_i_sparse1.size())
            next_ind_req1 = ext_dist.dist.all_to_all_single(next_output1, next_lS_i_sparse1, async_op=True)

        if cur_data[1]:
            # top mlp
            # with torch.autograd.profiler.record_function('Prof_top_mlp_forward1'):
            p1 = self.apply_mlp(z1, self.top_l)
            with torch.autograd.profiler.record_function('Prof_all_gather_t1'):
                T_test_req1.wait()

            # clamp output if needed
            if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
                z1 = torch.clamp(
                    p1, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
                )
            else:
                z1 = p1
            z1 = z1.float()

        if next_data[1]:
            next_ind_req1.wait()
            next_lS_i_sparse1 = next_output1.reshape(ext_dist.my_size, -1)
            next_g_i_sparse1 = [next_lS_i_sparse1[:, i * next_batch_size1:(i + 1) * next_batch_size1].reshape(-1).contiguous() for i in range(len(self.local_ln_emb_sparse))]
            next_lS_i_sparse1 = torch.cat(next_g_i_sparse1)
            next_data[1]['lS_i_dense_res'] = next_lS_i_sparse1

        if cur_data[1]:
            Z_test_req1, Z_test1 = ext_dist.all_gather_validation(z1, None)

        if next_data[0]:
            next_ly_sparse0 = self.apply_emb(next_batch_size0*ext_dist.my_size, next_data[0]['lS_i_dense_res'], self.emb_sparse)
            next_data[0]['ly_sparse'] = next_ly_sparse0

        if cur_data[1]:
            Z_test_req1.wait()


        if cur_data[1]:
            return (Z_test0, T_test0), (Z_test1, T_test1)
        else:
            return (Z_test0, T_test0), None



if __name__ == "__main__":
    # the reference implementation doesn't clear the cache currently
    # but the submissions are required to do that
    mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)

    mlperf_logger.log_start(key=mlperf_logger.constants.INIT_START, log_all_ranks=True)

    ### import packages ###
    import sys
    import os
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed run
    parser.add_argument("--dist-backend", type=str, default="ccl")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=-1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--profiling-start-iter", type=int, default=50)
    parser.add_argument("--profiling-num-iters", type=int, default=100)
    # store/load model
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # embedding table is sparse table only if sparse_dense_boundary >= 2048
    parser.add_argument("--sparse-dense-boundary", type=int, default=2048)
    # bf16 option
    parser.add_argument("--bf16", action='store_true', default=False)
    # lamb
    parser.add_argument("--optimizer", type=int, default=0, help='optimizer:[0:sgd, 1:lamb/sgd, 2:adagrad, 3:sparseadam]')
    parser.add_argument("--lamblr", type=float, default=0.01, help='lr for lamb')
    parser.add_argument("--use-hybridparallel-dataset", action="store_true", default=False)
    parser.add_argument("--ipex-interaction", action="store_true", default=False)
    parser.add_argument("--ipex-merged-emb", action="store_true", default=False)
    parser.add_argument("--use-ipex", action="store_true", default=False)
    parser.add_argument("--ddp-top-mlp", action="store_true", default=False)
    parser.add_argument("--ddp-bot-mlp", action="store_true", default=False)
    parser.add_argument("--hybrid-gradient-emb", action="store_true", default=False)
    parser.add_argument("--padding-last-test-batch", action="store_true", default=False)
    parser.add_argument("--enable-mlp-fusion", action="store_true", default=False)
    parser.add_argument("--allreduce-wait", action="store_true", default=False)
    args = parser.parse_args()

    if args.ipex_merged_emb and args.ipex_interaction:
        args.use_ipex =  True
    if args.use_ipex:
       import intel_extension_for_pytorch as ipex
       import intel_extension_for_pytorch._C as core

    ext_dist.init_distributed(backend=args.dist_backend)
    if ext_dist.my_size > 1:
       assert(args.hybrid_gradient_emb)

    dist_master_rank = (ext_dist.my_size > 1 and ext_dist.dist.get_rank() == 0)
    if args.mlperf_logging and dist_master_rank:
        print('command line args: ', json.dumps(vars(args)))
    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers
    if (args.mini_batch_size % ext_dist.my_size !=0 or args.test_mini_batch_size % ext_dist.my_size != 0):
        print("Either test minibatch (%d) or train minibatch (%d) does not split across %d ranks" % (args.test_mini_batch_size, args.mini_batch_size, ext_dist.my_size))
        sys.exit(1)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        if ext_dist.my_size > 1:
            ngpus = torch.cuda.device_count()  # 1
            if ext_dist.my_local_size > torch.cuda.device_count():
                print("Not sufficient GPUs available... local_size = %d, ngpus = %d" % (ext_dist.my_local_size, ngpus))
                sys.exit(1)
            ngpus = 1
            device = torch.device("cuda", ext_dist.my_local_rank)
        else:
            device = torch.device("cuda", 0)
            ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data


    if (args.data_generation == "dataset"):
        train_data, train_ld, test_data, test_ld = \
            dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
        ln_bot[0] = m_den

    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, (X, lS_i, T) in enumerate(train_ld):
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            '''
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            '''
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    print('Creating the model...', flush=True)
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        sparse_dense_boundary=args.sparse_dense_boundary,
        hybrid_gradient_emb=args.hybrid_gradient_emb,
        bf16 = args.bf16,
    )
    #print(dlrm)
    print('Model created!')
    if args.ipex_merged_emb and args.use_ipex:
        if args.hybrid_gradient_emb or ext_dist.my_size > 1:
            dlrm.emb_sparse = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(dlrm.emb_sparse, lr=args.learning_rate/ext_dist.my_size)
            dlrm.emb_dense = ipex.nn.modules.MergedEmbeddingBag.from_embeddingbag_list(dlrm.emb_dense)
            dlrm.need_linearize_indices_and_offsets = torch.BoolTensor([False])
        else:
            dlrm.emb_l = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(dlrm.emb_l, lr=args.learning_rate)
            dlrm.need_linearize_indices_and_offsets = torch.BoolTensor([False])
    print("#####afte merged", dlrm)
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)
    if args.allreduce_wait:
        dlrm.emb_dense_ar_req.wait = True
        dlrm.top_mlp_ar_req.wait = True
        dlrm.bot_mlp_ar_req.wait = True
    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

    def indices_hybrid_wrap(lS_i, last_padding_size=0):
        with torch.autograd.profiler.record_function("indices_hybrid_wrap:__concat__"):
            if last_padding_size > 0:
                import torch.nn.functional as F
                lS_i = F.pad(input=lS_i, pad=(0, last_padding_size), mode='constant', value=0)
            lS_i_sparse = [lS_i[i] for i in dlrm.ln_emb_sparse]
            lS_i_dense = [lS_i[i] for i in dlrm.ln_emb_dense]
            if not args.ipex_merged_emb:
                return (lS_i_sparse, lS_i_dense)
            if len(lS_i_sparse) > 1:
                lS_i_sparse_tensor = torch.cat(lS_i_sparse)
            else:
                lS_i_sparse_tensor = lS_i_sparse[0]
            if len(lS_i_dense) > 1:
                lS_i_dense_tensor = torch.cat(lS_i_dense)
            else:
                lS_i_dense_tensor = lS_i_dense[0]
            return (lS_i_sparse_tensor, lS_i_dense_tensor)

    buffer_num = 20
    if ext_dist.my_size > 1:
        buffer_num = 1
    data_buffer = buffer_num * [None]
    data_iter = iter(train_ld)
    buffer_num = buffer_num if buffer_num <= nbatches else nbatches
    data_load_begin = time.time()
    def load_data(data_iter, buffer_num):
        with torch.autograd.profiler.record_function('load_data'):
            for d in range(buffer_num):
                (X, lS_i, T) = next(data_iter)
                (lS_i_sparse_tensor, lS_i_dense_tensor) = indices_hybrid_wrap(lS_i)
                data_buffer[d] = (X.bfloat16(), T, lS_i_sparse_tensor, lS_i_dense_tensor)
    load_data(data_iter, buffer_num)
    print(buffer_num, ": data item loaded, data_load_time is {:.6f}s".format(time.time() - data_load_begin))

    if not args.inference_only:
        # specify the optimizer algorithm
        #optimizer_list = ([[torch.optim.SGD], ([Lamb, False], torch.optim.SGD),
        #                   torch.optim.Adagrad, ([torch.optim.Adam, None], torch.optim.SparseAdam)],
        #                  [[torch.optim.SGD], ([Lamb, True], torch.optim.SGD)])
        #optimizers = optimizer_list[args.bf16][args.optimizer]
        #print('Chosen optimizer(s): %s' % str(optimizers))
        #if ext_dist.my_size == 1:
        #    if len(optimizers) == 1:
        #        optimizer = optimizers[0](dlrm.parameters(), lr=args.learning_rate)
        #    else:
        #        optimizer_dense = optimizers[0][0]([
        #            {"params": dlrm.bot_l.parameters(), "lr": args.learning_rate},
        #            {"params": dlrm.top_l.parameters(), "lr": args.learning_rate}
        #        ], lr=args.learning_rate)
        #        if optimizers[0][1] is not None:
        #            optimizer_dense.set_bf16(optimizers[0][1])
        #        optimizer_sparse = optimizers[1]([
        #            {"params": [p for emb in dlrm.emb_l for p in emb.parameters()], "lr": args.learning_rate},
        #        ], lr=args.learning_rate)
        #        optimizer = (optimizer_dense, optimizer_sparse)
        #else:
        #    if len(optimizers) == 1:
        #        optimizer = optimizers[0]([
        #            {"params": [p for emb in dlrm.emb_sparse for p in emb.parameters()],
        #             "lr": args.learning_rate / ext_dist.my_size},
        #            {"params": [p for emb in dlrm.emb_dense for p in emb.parameters()], "lr": args.learning_rate},
        #            {"params": dlrm.bot_l.parameters(), "lr": args.learning_rate},
        #            {"params": dlrm.top_l.parameters(), "lr": args.learning_rate}
        #        ], lr=args.learning_rate)
        #    else:
        #        optimizer_dense = optimizers[0][0]([
        #            {"params": [p for emb in dlrm.emb_dense for p in emb.parameters()], "lr": args.learning_rate},
        #            {"params": dlrm.bot_l.parameters(), "lr": args.learning_rate},
        #            {"params": dlrm.top_l.parameters(), "lr": args.learning_rate}
        #        ], lr=args.lamblr, bf16=args.bf16)
        #        optimizer_sparse = optimizers[1]([
        #            {"params": [p for emb in dlrm.emb_sparse for p in emb.parameters()],
        #             "lr": args.learning_rate / ext_dist.my_size},
        #        ], lr=args.learning_rate)
        #        optimizer = (optimizer_dense, optimizer_sparse)
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            #"rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
            "adagrad": torch.optim.Adagrad,
        }
        #for name, p in dlrm.named_parameters():
        #    print(name)
        optimizers = []
        parameters_mlp = (
             [
                {
                    "params": dlrm.bot_l.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": dlrm.top_l.parameters(),
                    "lr": args.learning_rate,
                },
            ]
        )
        optimizer_mlp = opts["sgd"](parameters_mlp, lr=args.learning_rate)
        optimizers.append(optimizer_mlp)
        if not args.ipex_merged_emb:#aten::EmbeddingBag
            if args.hybrid_gradient_emb:
                parameters_emb = (
                        [
                            {
                                'params': dlrm.emb_dense.parameters(),
                                'lr': args.learning_rate
                            },
                            {
                                'params': dlrm.emb_sparse.parameters(),
                                'lr': args.learning_rate if ext_dist.my_size == 1 else args.learning_rate/ext_dist.my_size
                            }
                        ]
                    )
            else:
                parameters_emb = (
                        [
                            {
                                'params': dlrm.emb_l.parameters(),#all embeddingbag is sparse
                                'lr': args.learning_rate if ext_dist.my_size == 1 else args.learning_rate/ext_dist.my_size

                            }
                        ]
                    )
        else:#MergedEmbeddingBag
            if args.hybrid_gradient_emb:
                #only  need to use optimizer for dense embeddingbag
                #for sparse the MergedEmbeddingBagSGD has fused SGD optimizer
                parameters_emb = (
                        [
                            {
                                'params': dlrm.emb_dense.parameters(),
                                'lr': args.learning_rate

                            }
                        ]
                     )
            #else: all sparsed table will be a MergedEmbeddingBagSGD
        if args.hybrid_gradient_emb or not  args.ipex_merged_emb:
            emb_optimizer = opts["sgd"](parameters_emb, lr=args.learning_rate)
            optimizers.append(emb_optimizer)
        lr_schedulers =[]
        for optimizer in optimizers:
            scheduler = LRPolicyScheduler(
                optimizer,
                args.lr_num_warmup_steps,
                args.lr_decay_start_step,
                args.lr_num_decay_steps)
            lr_schedulers.append(scheduler)

    #pdb.set_trace()
    print("##########before optimizer")
    if args.use_ipex:
        dlrm, optimizers = ipex.optimize(dlrm, dtype=torch.bfloat16 if args.bf16 else torch.float32, optimizer=optimizers, inplace=True)
        #(X, _, lS_i_sparse_tensor, lS_i_dense_tensor) = data_buffer[0];
        #dlrm, optimizers = ipex.optimize(dlrm, dtype=torch.bfloat16, optimizer=optimizers, inplace=True,
        #    auto_kernel_selection=True, sample_input=(X, lS_i_sparse_tensor, lS_i_dense_tensor))
    print("##########finished optimizer")
    if args.bf16:
        if not args.inference_only and args.use_ipex:
            if args.ipex_merged_emb:
               if args.hybrid_gradient_emb and args.ipex_merged_emb:
                   dlrm.emb_sparse.to_bfloat16_train()
               elif args.ipex_merged_emb:
                   dlrm.emb_l.to_bfloat16_train()
        if args.use_ipex:
            for i in range(len(dlrm.top_l)):
                if isinstance(dlrm.top_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                    if isinstance(dlrm.top_l[i+1], torch.nn.ReLU):
                        dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'relu')
                    else:
                        dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'sigmoid')
                    dlrm.top_l[i + 1] = torch.nn.Identity()
            for i in range(len(dlrm.bot_l)):
                if isinstance(dlrm.bot_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                    if isinstance(dlrm.bot_l[i+1], torch.nn.ReLU):
                        dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'relu')
                    else:
                        dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'sigmoid')
                    dlrm.bot_l[i + 1] = torch.nn.Identity()
            if args.enable_mlp_fusion:
                dlrm.bot_l = ipex.nn.modules.BotMLP(dlrm.bot_l)
                dlrm.top_l = ipex.nn.modules.TopMLP(dlrm.top_l)
    flat_grads = None
    grad_dtype = torch.float32
    if ext_dist.my_size > 1:
        if args.ddp_top_mlp:
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, gradient_as_bucket_view=True, bucket_cap_mb=4)
        else:
            sz = 0
            szs = []
            for p in dlrm.top_l.parameters():
                sz += p.numel()
                szs.append(p.numel())
                grad_dtype = p.dtype
            flat_grads = torch.zeros([sz], dtype=grad_dtype)
            grads = flat_grads.split(szs)
            for p, g in zip(dlrm.top_l.parameters(), grads):
                p.grad = g.view_as(p.data)
                dlrm.top_mlp_flat_grads = flat_grads
                dlrm.top_mlp_ar_req.tensor = dlrm.top_mlp_flat_grads
            print("Top mlp weights totoal elements: {} every weight shape:{}".format(sz, szs))
        if args.ddp_bot_mlp:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
        else:
            sz = 0
            szs = []
            for p in dlrm.bot_l.parameters():
                sz += p.numel()
                szs.append(p.numel())
                grad_dtype = p.dtype
            flat_grads = torch.zeros([sz], dtype=grad_dtype)
            grads = flat_grads.split(szs)
            for p, g in zip(dlrm.bot_l.parameters(), grads):
                p.grad = g.view_as(p.data)
                dlrm.bot_mlp_flat_grads = flat_grads
                dlrm.bot_mlp_ar_req.tensor = dlrm.bot_mlp_flat_grads
            print("Bot mlp weights totoal elements: {} every weight shape:{}".format(sz, szs))
        if not args.ipex_merged_emb:
            for i in range(len(dlrm.emb_dense)):
                dlrm.emb_dense[i] = ext_dist.DDP(dlrm.emb_dense[i])
        else:
            sz = 0
            szs = []
            for p in dlrm.emb_dense.weights:
                sz += p.numel()
                szs.append(p.numel())
                grad_dtype = p.dtype
                torch.distributed.broadcast(p.data, 0)
            flat_grads = torch.zeros([sz], dtype=grad_dtype)
            grads = flat_grads.split(szs)
            for p, g in zip(dlrm.emb_dense.weights, grads):
                p.grad = g.view_as(p.data)
            dlrm.emb_dense_flat_grads = flat_grads
            dlrm.emb_dense_ar_req.tensor = dlrm.emb_dense_flat_grads
            print("Dense emb weights totoal elements: {} every weight shape:{}".format(sz, szs))

    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def loss_fn_wrap(Z, T, use_gpu, device):
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0
    if args.enable_profiling:
        nbatches = args.test_freq * 2
    should_test = ((args.test_freq > 0) and (args.data_generation == "dataset") and not args.inference_only)
    #should_test = ((args.test_freq > 0) and (args.data_generation == "dataset") and not args.inference_only and not args.enable_profiling)
    t_test_data_load = time.time()
    if should_test:
      test_data_buffer = nbatches_test * [None]
      i = 0
      while i < nbatches_test:
          (X_test, lS_i_test, T_test) = test_data[i]
          last_padding_size = 0
          if args.padding_last_test_batch and X_test.size(0) < args.test_mini_batch_size:
                last_padding_size = args.test_mini_batch_size - X_test.size(0)
                import torch.nn.functional as F
                X_test = F.pad(input=X_test, pad=(0, 0, 0, last_padding_size), mode='constant', value=0)
          (lS_i_test_sparse_tensor, lS_i_test_dense_tensor) = indices_hybrid_wrap(lS_i_test, last_padding_size)
          test_data_buffer[i] = (X_test.bfloat16(), T_test, lS_i_test_sparse_tensor, lS_i_test_dense_tensor)
          i += 1
    print("Dataloader time for test dataset ={} ms".format(1000*(time.time()- t_test_data_load)))
    mlperf_logger.mlperf_submission_log('dlrm')
    mlperf_logger.log_event(key=mlperf_logger.constants.SEED, value=args.numpy_rand_seed)
    mlperf_logger.log_event(key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.mini_batch_size)

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )

    ext_dist.barrier()
    mlperf_logger.barrier()
    mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
    mlperf_logger.barrier()
    mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
    mlperf_logger.barrier()

    #print("time/loss/accuracy (if enabled):")

    # LR is logged twice for now because of a compliance checker bug
    mlperf_logger.log_event(key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate)
    mlperf_logger.log_event(key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
                            value=args.lr_num_warmup_steps)

    # use logging keys from the official HP table and not from the logging library
    mlperf_logger.log_event(key='sgd_opt_base_learning_rate', value=args.learning_rate)
    mlperf_logger.log_event(key='lr_decay_start_steps', value=args.lr_decay_start_step)
    mlperf_logger.log_event(key='sgd_opt_learning_rate_decay_steps', value=args.lr_num_decay_steps)
    mlperf_logger.log_event(key='sgd_opt_learning_rate_decay_poly_power', value=2)

    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        if ext_dist.my_size > 1:
            rank = ext_dist.dist.get_rank()
            prof.export_chrome_trace("dlrm_training_trace_rank_{}_step{}.json".format(rank,str(prof.step_num)))
        else:
            rank = 0
            prof.export_chrome_trace("dlrm_training_trace_step_{}.json".format(str(prof.step_num)))

        file_prefix = "%s/dlrm_s_pytorch_r%d" % (".", rank)
        #with open("dlrm_s_pytorch.prof", "w") as prof_f:
        with open("%s.prof" % file_prefix, "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))

    def pack_test_data(output, test_data_buffer, start_iter, step_iter):
        for j in range(step_iter):
            if start_iter + j < len(test_data_buffer):
                X_test, T_test, lS_i_test_sparse_tensor, lS_i_test_dense_tensor = test_data_buffer[start_iter + j]
                output[j] = {'dense_x': X_test, 'T_test': T_test, 'lS_i_sparse': lS_i_test_sparse_tensor,
                    'lS_i_dense': lS_i_test_dense_tensor}
            else:
                output[j] = None

    #start_time = time_wrap(use_gpu)
    #with torch.autograd.profiler.profile(enabled=args.enable_profiling, use_cuda=use_gpu, record_shapes=True) as prof:
    wait_it = 0
    warmup_it = 400
    active_it = 20
    lrs =[]
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=wait_it,
            warmup=warmup_it,
            active=active_it),
        on_trace_ready=trace_handler
        ) as prof:
        while k < args.nepochs:
            mlperf_logger.barrier()
            mlperf_logger.log_start(key=mlperf_logger.constants.BLOCK_START,
                                    metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1),
                                              mlperf_logger.constants.EPOCH_COUNT: 1})
            mlperf_logger.barrier()
            mlperf_logger.log_start(key=mlperf_logger.constants.EPOCH_START,
                                    metadata={mlperf_logger.constants.EPOCH_NUM: k + 1})

            #if k < skip_upto_epoch:
            #    continue

            #accum_time_begin = time_wrap(use_gpu)

            t_bwd_total = 0
            t_fwd_total = 0
            t_opt_total = 0
            t_allreduce_total = 0


            j = 0
            cur_iter = 0
            #for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
            while j < nbatches:
                '''
                if j == 0 and args.save_onnx:
                    (X_onnx, lS_o_onnx, lS_i_onnx) = (X, lS_o, lS_i)
                if j < skip_upto_batch:
                    continue
                '''

                for d in range(buffer_num):
                    if args.print_time:
                        t1 = time_wrap(use_gpu)
                    (X, T, lS_i_sparse_tensor, lS_i_dense_tensor) = data_buffer[d]
                    cur_iter += 1
                    '''
                    # debug prints
                    print("input and targets")
                    print(X.detach().cpu().numpy())
                    print([np.diff(S_o.detach().cpu().tolist()
                       + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
                    print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
                    print(T.detach().cpu().numpy())
                    '''
                    if args.hybrid_gradient_emb or ext_dist.my_size > 1:
                        if isinstance(dlrm.emb_sparse, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                            dlrm.emb_sparse.sgd_args = dlrm.emb_sparse.sgd_args._replace(lr=lr_schedulers[0].get_last_lr()[0]/ext_dist.my_size)
                        else:
                            if args.use_ipex and isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                                dlrm.emb_l.sgd_args = dlrm.emb_l.sgd_args._replace(lr=lr_schedulers[0].get_last_lr()[0])
                    t_fwd_beg = time_wrap(use_gpu)
                    with torch.cpu.amp.autocast(enabled=args.bf16), torch.autograd.profiler.record_function('Prof_dlrm_forward'):
                        # forward pass
                        Z = dlrm(X, lS_i_sparse_tensor, lS_i_dense_tensor).float()
                    # loss
                    with torch.autograd.profiler.record_function('Prof_loss'):
                        E = loss_fn_wrap(Z, T, use_gpu, device)
                    if not args.inference_only:
                        if ext_dist.my_size == 1:#zero_grad will be overlapped for multi-ranks.
                           for optimizer in optimizers:
                               optimizer.zero_grad()

                    t_fwd_total += (time_wrap(use_gpu) - t_fwd_beg)

                    '''
                    # debug prints
                    print("output and loss")
                    print(Z.detach().cpu().numpy())
                    print(E.detach().cpu().numpy())
                    '''
                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    S = Z.detach().cpu().numpy()  # numpy array
                    T = T.detach().cpu().numpy()  # numpy array
                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    if not args.inference_only:
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        #if args.optimizer == 1 or args.optimizer == 3:
                        #    optimizer_dense.zero_grad()
                        #    optimizer_sparse.zero_grad()
                        #backward pass
                        t_bwd_beg = time_wrap(use_gpu)
                        E.backward()
                        t_bwd_total += (time_wrap(use_gpu) - t_bwd_beg)
                        dense_emb_req = None
                        if ext_dist.my_size > 1 and args.hybrid_gradient_emb:
                            t_allreduce_beg = time_wrap(use_gpu)
                            if dlrm.emb_dense_flat_grads is None:
                            #    dlrm.emb_dense_flat_grads.div_(ext_dist.my_size)
                            #    dlrm.emb_dense_ar_req.request = ext_dist.dist.all_reduce(dlrm.emb_dense_flat_grads, async_op=True)
                            #else:
                                for weight in dlrm.emb_dense.weights:
                                    weight.grad.div_(ext_dist.my_size)
                                    ext_dist.dist.all_reduce(weight.grad)
                            t_allreduce_total += (time_wrap(use_gpu) - t_allreduce_beg)
                        #print("#####################")
                        #for i in range(len(dlrm.emb_dense.weights)):
                        #    print(dlrm.emb_dense.weights[i].grad)
                        # debug prints (check gradient norm)
                        # for l in mlp.layers:
                        #     if hasattr(l, 'weight'):
                        #          print(l.weight.grad.norm().item())

                        t_opt_beg = time_wrap(use_gpu)
                        # optimizer
                        #optimizer = optimizers[0]
                        #dense_emb_optimizer = optimizers[1]
                        #if args.optimizer == 1 or args.optimizer == 3:
                        #   with torch.autograd.profiler.record_function("optimizer_dense:step"):
                        #        optimizer_dense.step()
                        #    with torch.autograd.profiler.record_function("optimizer_sparse:step"):
                        #        optimizer_sparse.step()
                        #else:
                        if ext_dist.my_size > 1 and not args.ddp_bot_mlp and not dlrm.bot_mlp_ar_req.wait:
                            dlrm.bot_mlp_ar_req.request.wait()
                        if ext_dist.my_size > 1 and not args.ddp_top_mlp and not dlrm.top_mlp_ar_req.wait:
                            dlrm.top_mlp_ar_req.request.wait()
                        if args.use_ipex and args.hybrid_gradient_emb and ext_dist.my_size > 1:
                            optimizer_mlp = optimizers[0]
                            optimizer_mlp.step()
                            schduler_mlp = lr_schedulers[0]
                            schduler_mlp.step()
                            dlrm.emb_dense_ar_req.request.wait()
                            if should_test and (cur_iter % args.test_freq == 0):
                                 optimizers[1].step()
                                 lr_schedulers[1].step()
                                 print("should test after allreduce dense_emb_optimizer.step/lr_schedulers[1].step")
                        else:
                                for optimizer in optimizers:
                                    optimizer.step()
                                for lr_scheduler in lr_schedulers:
                                    lr_scheduler.step()

                        t_opt_total += (time_wrap(use_gpu) - t_opt_beg)

                        if args.enable_profiling:
                            prof.step()
                            if cur_iter >=  wait_it + warmup_it + active_it:
                                break

                    total_accu += A
                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    if args.print_time:
                        total_time += time_wrap(use_gpu) - t1
                    should_print = ((cur_iter % args.print_freq == 0) or (cur_iter == nbatches))
                    # print time, loss and accuracy
                    if should_print:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        gFwdT = 1000.0 * t_fwd_total / total_iter
                        gBwdT = 1000.0 * t_bwd_total / total_iter
                        gOptT = 1000.0 * t_opt_total / total_iter
                        gAllreduceT = 1000.0 * t_allreduce_total / total_iter

                        gA = total_accu / total_samp
                        total_accu = 0

                        gL = total_loss / total_samp
                        total_loss = 0

                        str_run_type = "inference" if args.inference_only else "training"
                        if ext_dist.my_size == 1 or dist_master_rank:
                            print(
                                "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, fwd: {:.2f} ms/it, bwd: {:.2f} ms/it, optimizier: {:.2f} ms/it, EMB-allreduce: {:.2f} ms/it,".format(
                                    str_run_type, cur_iter, nbatches, k, gT, gFwdT, gBwdT, gOptT, gAllreduceT
                                #"Finished {} it {}/{} of epoch {}, {:.2f} ms/it, EMB-allreduce: {:.2f} ms/it,".format(
                                #    str_run_type, cur_iter, nbatches, k, gT, gAllreduceT
                                )
                                + "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                            )
                        # Uncomment the line below to print out the total time with overhead
                        # print("Accumulated time so far: {}" \
                        # .format(time_wrap(use_gpu) - accum_time_begin))
                        total_iter = 0
                        total_samp = 0
                        t_bwd_total = 0
                        t_fwd_total = 0
                        t_opt_total = 0
                        t_allreduce_total = 0

                    # testing
                    if should_test and (cur_iter % args.test_freq == 0):
                        dlrm.start_train_iteration = True
                        epoch_num_float = cur_iter / nbatches + k + 1
                        mlperf_logger.barrier()
                        mlperf_logger.log_start(key=mlperf_logger.constants.EVAL_START,
                                            metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})

                        test_accu = 0
                        test_loss = 0
                        test_samp = 0

                        accum_test_time_begin = time_wrap(use_gpu)
                        if args.mlperf_logging:
                            scores = []
                            targets = []

                        i = 0
                        if ext_dist.my_size > 1:
                            n_test_iter = 2     # exec 2 iter per validation
                            cur_data = [dict() for _ in range(n_test_iter)]
                            next_data = [dict() for _ in range(n_test_iter)]
                            n = nbatches_test / n_test_iter
                            if nbatches_test % n_test_iter:
                                n += 1
                            while i < n:

                                # load 2 iter
                                if not i:
                                    pack_test_data(cur_data, test_data_buffer, n_test_iter*i, n_test_iter)
                                else:
                                    for j in range(n_test_iter):
                                        cur_data[j] = next_data[j]
                                # preload 2 iter
                                pack_test_data(next_data, test_data_buffer, n_test_iter*(i+1), n_test_iter)

                                pre = dlrm.validation(cur_data, next_data)

                                for j in range(n_test_iter):
                                    if pre[j]:
                                        Z_test, T_test = pre[j]
                                        S_test = Z_test.detach().cpu().float()  # numpy array
                                        T_test = T_test.detach().cpu().float()  # numpy array
                                        if args.padding_last_test_batch and T_test.size(0) < args.test_mini_batch_size:
                                            S_test = S_test[:T_test.size(0)]
                                        scores.append(S_test)
                                        targets.append(T_test)
                                i += 1
                        else:
                            while i < nbatches_test:
                                (X_test, T_test, lS_i_test_sparse_tensor, lS_i_test_dense_tensor) = test_data_buffer[i]

                                # forward pass
                                with torch.cpu.amp.autocast(enabled=args.bf16):
                                    Z_test = dlrm(
                                        X_test, lS_i_test_sparse_tensor, lS_i_test_dense_tensor, False
                                    ).float()
                                i += 1
                                if args.mlperf_logging:
                                    if ext_dist.my_size > 1:
                                        Z_test = ext_dist.all_gather(Z_test, None)
                                        T_test = ext_dist.all_gather(T_test, None)
                                    S_test = Z_test.detach().cpu().float()  # numpy array
                                    T_test = T_test.detach().cpu().float()  # numpy array
                                    if args.padding_last_test_batch and T_test.size(0) < args.test_mini_batch_size:
                                        S_test = S_test[:T_test.size(0)]
                                    scores.append(S_test)
                                    targets.append(T_test)
                                else:
                                    # loss
                                    E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

                                    # compute loss and accuracy
                                    L_test = E_test.detach().cpu().numpy()  # numpy array
                                    S_test = Z_test.detach().cpu().numpy()  # numpy array
                                    if args.padding_last_test_batch and T_test.size(0) < args.test_mini_batch_size:
                                        S_test = S_test[:T_test.size(0)]
                                    T_test = T_test.detach().cpu().numpy()  # numpy array
                                    mbs_test = T_test.shape[0]  # = mini_batch_size except last
                                    A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                                    test_accu += A_test
                                    test_loss += L_test * mbs_test
                                    test_samp += mbs_test

                        if args.mlperf_logging:
                            scores = torch.cat(scores, 0).reshape(-1)
                            targets = torch.cat(targets, 0).reshape(-1)
                            validation_results = {}
                            #validation_results['roc_auc'], validation_results['loss'], validation_results['accuracy'] = \
                            validation_results['roc_auc'], _, validation_results['accuracy'] = \
                                    core.roc_auc_score_all(targets, scores)
                            gA_test = validation_results['accuracy']
                            is_best = validation_results['roc_auc'] > best_auc_test
                            if is_best:
                                best_auc_test = validation_results['roc_auc']
                            if gA_test > best_gA_test:
                                best_gA_test = gA_test

                            mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_ACCURACY,
                                                value=float(validation_results['roc_auc']),
                                                metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})
                            if ext_dist.my_size == 1 or dist_master_rank:
                                print(
                                    "Testing at - {}/{} of epoch {},".format(cur_iter, nbatches, k)
                                    #+ " loss {:.6f},".format(validation_results['loss'])
                                    + " auc {:.6f}, best auc {:.6f},".format(
                                        validation_results['roc_auc'],
                                        best_auc_test
                                    )
                                    + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                                        validation_results['accuracy'] * 100,
                                        best_gA_test * 100
                                    )
                                )

                            mlperf_logger.barrier()
                            mlperf_logger.log_end(key=mlperf_logger.constants.EVAL_STOP,
                                          metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})
                            if ((args.mlperf_auc_threshold > 0)
                                and (best_auc_test > args.mlperf_auc_threshold)):
                                print("MLPerf testing auc threshold "
                                  + str(args.mlperf_auc_threshold)
                                  + " reached, stop training")
                                mlperf_logger.barrier()
                                mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                                                  metadata={mlperf_logger.constants.STATUS: mlperf_logger.constants.SUCCESS})
                                break
                        else:
                            gA_test = test_accu / test_samp
                            gL_test = test_loss / test_samp
                            is_best = gA_test > best_gA_test
                            if is_best:
                                best_gA_test = gA_test
                            print(
                                "Testing at - {}/{} of epoch {},".format(cur_iter, nbatches, 0)
                                + " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                    gL_test, gA_test * 100, best_gA_test * 100
                                )
                            )

                        if is_best and not (args.save_model == ""):
                            print("Saving model to {}".format(args.save_model))
                            torch.save(
                                {
                                        "epoch": k,
                                        "nepochs": args.nepochs,
                                        "nbatches": nbatches,
                                        "nbatches_test": nbatches_test,
                                        "iter": cur_iter + 1,
                                        "state_dict": dlrm.state_dict(),
                                        "train_acc": gA,
                                        "train_loss": gL,
                                        "test_acc": gA_test,
                                        "test_loss": gL_test,
                                        "total_loss": total_loss,
                                        "total_accu": total_accu,
                                        "opt_state_dict": optimizer.state_dict(),
                                },
                                args.save_model,
                            )

                        # Uncomment the line below to print out the total time with overhead
                        if ext_dist.my_size == 1 or dist_master_rank:
                            print("Total test time for this group: {:.6f}s" \
                                .format(time_wrap(use_gpu) - accum_test_time_begin))

                if (best_auc_test > args.mlperf_auc_threshold):
                    break

                j += buffer_num
                if j >= nbatches:
                    break
                if ext_dist.my_size == 1:
                    buffer_num = buffer_num if (nbatches - j) > buffer_num else (nbatches - j)
                    load_data(data_iter, buffer_num)

            if args.mlperf_logging:
                mlperf_logger.barrier()
                mlperf_logger.log_end(key=mlperf_logger.constants.EPOCH_STOP,
                                      metadata={mlperf_logger.constants.EPOCH_NUM: k + 1})
                mlperf_logger.barrier()
                mlperf_logger.log_end(key=mlperf_logger.constants.BLOCK_STOP,
                                      metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: k + 1})
            k += 1  # nepochs
            if k < args.nepochs:
                data_iter = iter(train_ld)
                buffer_num = buffer_num if buffer_num <= nbatches else nbatches
                load_data(data_iter, buffer_num)

    if args.mlperf_logging and best_auc_test <= args.mlperf_auc_threshold:
        mlperf_logger.barrier
        mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                              metadata={mlperf_logger.constants.STATUS: mlperf_logger.constants.ABORTED})

    #print("Total used time: {:.6f} s".format(time_wrap(use_gpu) - start_time))
    print('DLRM training summary: best_auc_test = %.6f    ' % best_auc_test)


    '''
    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())


    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # export the model in onnx
    if args.save_onnx:
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        torch.onnx.export(
            dlrm, (X_onnx, lS_o_onnx, lS_i_onnx), dlrm_pytorch_onnx_file, verbose=True, use_external_data_format=True
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
    '''
