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

import re
import random
import time

from requests.adapters import ProxyError

import torch
from torch.utils.data import DataLoader as loader
from transformers import AutoTokenizer

from cloudtik.runtime.ai.modeling.transfer_learning.dataset import Dataset


class HuggingFaceDataset(Dataset):
    """
    Base class to represent Hugging Face Dataset
    """

    def __init__(self, dataset_dir, dataset_name=""):
        Dataset.__init__(self, dataset_dir, dataset_name)

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
        elif subset == 'train' and self.train_subset is not None:
            return next(iter(self._train_loader))
        elif subset == 'validation' and self.validation_subset is not None:
            return next(iter(self._validation_loader))
        elif subset == 'test' and self.test_subset is not None:
            return next(iter(self._test_loader))
        else:
            raise ValueError("Unable to return a batch, because the dataset or subset hasn't been defined.")

    def preprocess(
        self,
        model_name: str,
        batch_size: int = 32,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 64,
        **kwargs
    ) -> None:
        """
        Preprocess the textual dataset to apply padding, truncation and tokenize.

            Args:
                model_name (str): Name of the model to get a matching tokenizer.
                batch_size (int): Number of batches to split the data.
                padding (str): desired padding. (default: "max_length")
                max_length (int): desired max length. (default: 64)
                truncation (bool): Boolean specifying to truncate the word tokens to match with the
                longest sentence. (default: True)
                max_length (int): Maximum sequence length
            Raises:
                ValueError if data has already been preprocessed (or) non integer batch size given (or)
                given dataset hasn't been implemented into the API yet.
        """

        # Sanity checks
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be an positive integer")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        column_names = self._dataset.column_names

        # There must be at least one feature named 'label' in the self._dataset. The remaining features
        # become the text columns provided they contain only strings
        text_column_names = [col_name for col_name in column_names if col_name != 'label' and
                             all(isinstance(s, str) for s in self._dataset[col_name])]

        # Get the tokenizer
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ProxyError:
            print("Max retries reached. Sleeping for 10 sec...")
            time.sleep(10)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define a tokenize function to map the text to the tokenizer
        def tokenize_function(examples):
            # Define the tokenizer args, depending on number of text columns present in the dataset
            args = (examples[text_column_name] for text_column_name in text_column_names)

            result = self._tokenizer(*args, padding=padding, max_length=max_length, truncation=truncation)
            return result

        self._dataset = self._dataset.map(tokenize_function, batched=True)

        # Prepare the tokenized dataset in the format expected by model.
        # Remove the rest of the features from the tokenized dataset except 'label'
        self._dataset = self._dataset.remove_columns([col for col in column_names if col != 'label'])

        # Set format to torch
        self._dataset.set_format("torch")

        self._preprocessed = {
            'padding': padding,
            'truncation': truncation,
            'batch_size': batch_size,
        }
        self._make_data_loaders(batch_size=batch_size)
        print("Tokenized Dataset:", self._dataset)

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
        # Sanity checks
        if not (isinstance(train_pct, float) and isinstance(val_pct, float) and isinstance(test_pct, float)):
            raise ValueError("Percentage arguments must be floats.")

        if train_pct + val_pct + test_pct > 1.0:
            raise ValueError("Sum of percentage arguments must be less than or equal to 1.")

        self._validation_type = 'shuffle_split'

        # Calculating splits
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

        if self._preprocessed and 'batch_size' in self._preprocessed:
            self._make_data_loaders(batch_size=self._preprocessed['batch_size'], generator=generator)

        print("Dataset split into:")
        print("-------------------")
        print("{} train samples".format(train_size))
        print("{} test samples".format(test_size))
        print("{} validation samples".format(val_size))

    def _make_data_loaders(self, batch_size, generator=None):

        def seed_worker(worker_id):
            import numpy as np
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if self._validation_type == 'shuffle_split':
            self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=self._shuffle,
                                        num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)

            self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=self._shuffle,
                                             num_workers=self._num_workers, worker_init_fn=seed_worker,
                                             generator=generator)

            if self._test_indices:
                self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=self._shuffle,
                                           num_workers=self._num_workers, worker_init_fn=seed_worker,
                                           generator=generator)

        elif self._validation_type == 'defined_split':
            if 'train' in self._split:
                self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=self._shuffle,
                                            num_workers=self._num_workers, worker_init_fn=seed_worker,
                                            generator=generator)
            if 'test' in self._split:
                self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=self._shuffle,
                                           num_workers=self._num_workers, worker_init_fn=seed_worker,
                                           generator=generator)
            if 'validation' in self._split:
                self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=self._shuffle,
                                                 num_workers=self._num_workers, worker_init_fn=seed_worker,
                                                 generator=generator)
        elif self._validation_type is None:

            self._data_loader = loader(self._dataset, batch_size=batch_size, shuffle=self._shuffle,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator)

            self._train_loader = self._data_loader
            self._test_loader = self._data_loader
            self._validation_loader = self._data_loader

    def get_text(self, input_ids):
        """
        Helper function to decode the input_ids to text
        """
        decoded_text = []
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        if input_ids.ndim > 1:
            decoded_tokens = self._tokenizer.batch_decode(input_ids)
            for t in decoded_tokens:
                decoded_text.append(re.search(r'\[CLS\] (.*?) \[SEP\]', t).group(1))
        else:
            decoded_tokens = self._tokenizer.decode(input_ids)
            decoded_text.append(re.search(r'\[CLS\] (.*?) \[SEP\]', decoded_tokens).group(1))

        return decoded_text

    @property
    def train_subset(self):
        train_ds = None

        if self._validation_type == 'shuffle_split':
            train_ds = self._dataset.select(self._train_indices)
        elif self._validation_type == 'defined_split':
            if 'train' in self._split:
                train_ds = self._dataset.select(self._train_indices)
            else:
                raise ValueError("train split not specified")
        elif self._validation_type is None:
            train_ds = self._dataset

        return train_ds

    @property
    def test_subset(self):
        test_ds = None

        if self._validation_type == 'shuffle_split':
            if self._test_indices:
                test_ds = self._dataset.select(self._test_indices)
        elif self._validation_type == 'defined_split':
            if 'test' in self._split:
                test_ds = self._dataset.select(self._test_indices)
            else:
                raise ValueError("test split not specified")
        elif self._validation_type is None:
            test_ds = self._dataset

        return test_ds

    @property
    def validation_subset(self):
        validation_ds = None

        if self._validation_type == 'shuffle_split':
            validation_ds = self._dataset.select(self._validation_indices)
        elif self._validation_type == 'defined_split':
            if 'validation' in self._split:
                validation_ds = self._dataset.select(self._validation_indices)
            else:
                raise ValueError("validation split not specified")
        elif self._validation_type is None:
            validation_ds = self._dataset

        return validation_ds

    @property
    def train_loader(self):
        if self._train_loader:
            return self._train_loader
        else:
            raise ValueError("train split not specified")

    @property
    def test_loader(self):
        if self._test_loader:
            return self._test_loader
        else:
            raise ValueError("test split not specified")

    @property
    def validation_loader(self):
        if self._validation_loader:
            return self._validation_loader
        else:
            raise ValueError("validation split not specified")
