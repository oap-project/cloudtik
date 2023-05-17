#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
import random
import inspect
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader as loader

from ..dataset import Dataset


class PyTorchDataset(Dataset):
    """
    Base class to represent a PyTorch Dataset
    """

    def __init__(self, dataset_dir, dataset_name=""):
        """
        Class constructor
        """
        Dataset.__init__(self, dataset_dir, dataset_name)

    @property
    def train_subset(self):
        """
        A subset of the dataset used for training
        """
        return torch.utils.data.Subset(self._dataset, self._train_indices) if self._train_indices else None

    @property
    def validation_subset(self):
        """
        A subset of the dataset used for validation/evaluation
        """
        return torch.utils.data.Subset(self._dataset, self._validation_indices) if self._validation_indices else None

    @property
    def test_subset(self):
        """
        A subset of the dataset held out for final testing/evaluation
        """
        return torch.utils.data.Subset(self._dataset, self._test_indices) if self._test_indices else None

    @property
    def data_loader(self):
        """
        A data loader object corresponding to the dataset
        """
        return self._data_loader

    @property
    def train_loader(self):
        """
        A data loader object corresponding to the training subset
        """
        return self._train_loader

    @property
    def validation_loader(self):
        """
        A data loader object corresponding to the validation subset
        """
        return self._validation_loader

    @property
    def test_loader(self):
        """
        A data loader object corresponding to the test subset
        """
        return self._test_loader

    def get_batch(self, subset='all'):
        """
        Get a single batch of images and labels from the dataset.

            Args:
                subset (str): default "all", can also be "train", "validation", or "test"

            Returns:
                (examples, labels)

            Raises:
                ValueError if the dataset is not defined yet or the given subset is not valid
        """
        if subset == 'all' and self._dataset is not None:
            return next(iter(self._data_loader))
        elif subset == 'train' and self._train_loader is not None:
            return next(iter(self._train_loader))
        elif subset == 'validation' and self._validation_loader is not None:
            return next(iter(self._validation_loader))
        elif subset == 'test' and self._test_loader is not None:
            return next(iter(self._test_loader))
        else:
            raise ValueError("Unable to return a batch, because the dataset or subset hasn't been defined.")

    def shuffle_split(self, train_pct=.75, val_pct=.25, test_pct=0., shuffle_files=True, seed=None):
        """
        Randomly split the dataset into train, validation, and test subsets with a pseudo-random seed option.

            Args:
                train_pct (float): default .75, percentage of dataset to use for training
                val_pct (float):  default .25, percentage of dataset to use for validation
                test_pct (float): default 0.0, percentage of dataset to use for testing
                shuffle_files (bool): default True, optionally control whether shuffling occurs
                seed (None or int): default None, can be set for pseudo-randomization

            Raises:
                ValueError if percentage input args are not floats or sum to greater than 1
        """
        if not (isinstance(train_pct, float) and isinstance(val_pct, float) and isinstance(test_pct, float)):
            raise ValueError("Percentage arguments must be floats.")
        if train_pct + val_pct + test_pct > 1.0:
            raise ValueError("Sum of percentage arguments must be less than or equal to 1.")

        length = len(self._dataset)
        train_size = int(train_pct * length)
        val_size = int(val_pct * length)
        test_size = int(test_pct * length)
        generator = torch.Generator().manual_seed(seed) if seed else None
        if shuffle_files:
            dataset_indices = torch.randperm(length, generator=generator).tolist()
        else:
            dataset_indices = range(length)
        self._train_indices = dataset_indices[:train_size]
        self._validation_indices = dataset_indices[train_size:train_size + val_size]
        if test_pct:
            self._test_indices = dataset_indices[train_size + val_size:train_size + val_size + test_size]
        else:
            self._test_indices = None
        self._validation_type = 'shuffle_split'
        if self._preprocessed and 'batch_size' in self._preprocessed:
            self._make_data_loaders(batch_size=self._preprocessed['batch_size'], generator=generator)

    def _make_data_loaders(self, batch_size, generator=None):
        """Make data loaders for the whole dataset and the subsets that have indices defined"""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if self._dataset:
            self._data_loader = loader(self.dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)
        else:
            self._data_loader = None
        if self._train_indices:
            self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)
        else:
            self._train_loader = None
        if self._validation_indices:
            self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=False,
                                             num_workers=self._num_workers, worker_init_fn=seed_worker,
                                             generator=generator)
        else:
            self._validation_loader = None
        if self._test_indices:
            self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=False,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker,
                                       generator=generator)
        else:
            self._test_loader = None

    def preprocess(self, image_size='variable', batch_size=32, add_aug=None, **kwargs):
        """
        Preprocess the dataset to resize, normalize, and batch the images. Apply augmentation
        if specified.

            Args:
                image_size (int or 'variable'): desired square image size (if 'variable', does not alter image size)
                batch_size (int): desired batch size (default 32)
                add_aug (None or list[str]): Choice of augmentations (RandomHorizontalFlip, RandomRotation) to be
                                             applied during training
                kwargs: optional; additional keyword arguments for Resize and Normalize transforms
            Raises:
                ValueError if the dataset is not defined or has already been processed
        """
        # NOTE: Should this be part of init? If we get image_size and batch size during init,
        # then we don't need a separate call to preprocess.
        if not (self._dataset):
            raise ValueError("Unable to preprocess, because the dataset hasn't been defined.")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be an positive integer")

        if not image_size == 'variable' and not (isinstance(image_size, int) and image_size >= 1):
            raise ValueError("Input image_size must be either a positive int or 'variable'")

        # Get the user-specified keyword arguments
        resize_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(T.Resize).args}
        normalize_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(T.Normalize).args}

        def get_transform(image_size, add_aug):
            transforms = []
            if isinstance(image_size, int):
                transforms.append(T.Resize([image_size, image_size], **resize_args))
            if add_aug is not None:
                aug_dict = {'hflip': T.RandomHorizontalFlip(),
                            'rotate': T.RandomRotation(0.5)}
                aug_list = ['hflip', 'rotate']
                for option in add_aug:
                    if option not in aug_list:
                        raise ValueError("Unsupported augmentation for PyTorch:{}. \
                        Supported augmentations are {}".format(option, aug_list))
                    transforms.append(aug_dict[option])
            transforms.append(T.ToTensor())
            transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], **normalize_args))

            return T.Compose(transforms)

        self._dataset.transform = get_transform(image_size, add_aug)
        self._preprocessed = {'image_size': image_size, 'batch_size': batch_size}
        self._make_data_loaders(batch_size=batch_size)
