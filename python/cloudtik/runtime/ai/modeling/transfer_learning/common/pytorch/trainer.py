#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

import os

import torch
import torch.distributed as dist

from tqdm import tqdm
from random import Random
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


""" Dataset partitioning helper classes and methods """


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []

        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size):
    world_size = dist.get_world_size()
    bsz = int(batch_size / world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_loader = DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_loader, bsz


""" Distributed Torch helper classes """


class TrainArguments:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = dict(kwargs)


class Trainer:
    def __init__(self) -> None:
        pass

    def run(
            self,
            training_args: TrainArguments,
            master_addr: str,
            master_port: str,
            backend: str = 'ccl'
    ):
        self.setup(master_addr, master_port, backend)

        self._fit(training_args)

        self.cleanup()

    def _fit(self, training_args: TrainArguments):
        self._model = training_args.model
        self._optimizer = training_args.optimizer
        self._criterion = training_args.criterion

        if training_args.ipex:
            import intel_extension_for_pytorch as ipex
            self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)

        self._ddp_model = DDP(self._model)

        dataset = training_args.dataset
        batch_size = training_args.batch_size
        epochs = training_args.epochs

        dataloader, bsz = partition_dataset(dataset, batch_size)
        epoch_accuracies, epoch_losses = [], []

        # Since we are loading the model from disk, we have to set 'requires_grad'
        # to True for the optimizer to update the model parameters.
        for param in self._ddp_model.parameters():
            param.requires_grad = True

        if training_args.category == 'text_classification':
            for epoch in range(epochs):
                print(f'Epoch {epoch+1}/{epochs}')
                print('-' * 10)

                # Training phase
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data_batch in tqdm(dataloader):
                    inputs = {k: v for k, v in data_batch.items()
                              if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                    labels = data_batch['label']

                    # zero the parameter gradients
                    self._optimizer.zero_grad()

                    # Forward pass
                    outputs = self._ddp_model(**inputs)
                    loss = self._criterion(outputs.logits, labels)

                    # Backward pass
                    loss.backward()
                    self.average_gradients()
                    self._optimizer.step()

                    # Statistics
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    running_loss += torch.as_tensor(loss.item() * data_batch['input_ids'].size(0))
                    running_corrects += torch.sum(predictions == labels)

                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects / len(dataset)
                epoch_accuracies.append(epoch_acc)
                epoch_losses.append(epoch_loss)

                print("Loss: {}".format(epoch_loss))
                print("Acc: {}".format(epoch_acc))

            training_loss = epoch_losses[-1]
            training_acc = epoch_accuracies[-1]

            if dist.get_rank() == 0:
                print("Training loss:", training_loss)
                print("Training accuracy:", training_acc)
        elif training_args.category == 'image_classification':
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch + 1, epochs))

                running_loss = 0
                running_corrects = 0
                for data, target in tqdm(dataloader):
                    self._optimizer.zero_grad()
                    out = self._ddp_model(data)
                    loss = self._criterion(out, target)
                    loss.backward()
                    self.average_gradients()
                    self._optimizer.step()

                    # Statistics
                    preds = torch.argmax(out, dim=1)
                    running_loss += torch.as_tensor(loss.item() * data.size(0))
                    running_corrects += torch.sum(preds == target)

                # Collect all the running_loss and running_corrects tensors
                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects / len(dataset)
                epoch_accuracies.append(epoch_acc)
                epoch_losses.append(epoch_loss)

                print("Loss: {}".format(epoch_loss))
                print("Acc: {}".format(epoch_acc))

            training_loss = epoch_losses[-1]
            training_acc = epoch_accuracies[-1]

            if dist.get_rank() == 0:
                print("Training loss:", training_loss)
                print("Training accuracy:", training_acc)
        else:
            raise ValueError("PyTorch Distributed Training for {} is not implemented yet"
                             .format(training_args.category))

    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self._ddp_model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    @classmethod
    def setup(cls, master_addr: str, master_port: str, backend: str = 'gloo'):
        if dist.is_initialized():
            print("Process Group already initialized")
        else:
            # Usually the launcher will set the MASTER_ADDR and MASTER_PORT env
            if master_addr:
                os.environ['MASTER_ADDR'] = master_addr
            if master_addr:
                os.environ['MASTER_PORT'] = master_port
            os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')

            if backend == 'ccl':
                import oneccl_bindings_for_pytorch  # noqa # pylint: disable=unused-import

            dist.init_process_group(
                backend=backend,
                init_method='env://'
            )

    @classmethod
    def cleanup(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    @classmethod
    def load_objects(cls, objects_path):
        """
        Helper function to load saved dataset and model objects

        Args:
            objects_path (str): The shared data path to load objects.
        Returns:
            dict with loaded dataset and model objects
        """
        return torch.load(objects_path)
