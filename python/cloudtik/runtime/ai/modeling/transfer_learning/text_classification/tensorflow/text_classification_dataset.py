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

import os
import tensorflow as tf

from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.text_classification_dataset \
    import TextClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.datasets import DataDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.tensorflow.dataset import TensorflowDataset
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import read_json_file


class TensorflowTextClassificationDataset(TensorflowDataset, TextClassificationDataset):
    """
    A text classification dataset from the TensorFlow datasets catalog
    """
    def __init__(self, dataset_dir, dataset_name, split=["train"], shuffle_files=True, **kwargs):
        if not isinstance(split, list):
            raise ValueError("Value of split argument must be a list.")

        TextClassificationDataset.__init__(self, dataset_dir, dataset_name)

        this_dir = os.path.dirname(__file__)
        config_file = os.path.join(this_dir, "tf_text_classification_datasets.json")
        config_dict = read_json_file(config_file)
        available_datasets = list(config_dict.keys())
        if dataset_name not in available_datasets:
            raise ValueError("Dataset name is not supported. Choose from: {}".format(available_datasets))

        # as_supervised gives us the (input, label) structure that the model expects
        as_supervised = True

        # Glue datasets don't support as_supervised=True, so we need to set as_supervised=False, and then fix
        # the data format after loading
        if "glue" in dataset_name:
            as_supervised = False

        downloader = DataDownloader(dataset_name, dataset_dir=dataset_dir,
                                    source='tfds', as_supervised=as_supervised,
                                    shuffle_files=shuffle_files, with_info=True)
        data, self._info = downloader.download(split=split)

        # Since glue datasets don't support the supervised (input, label) structure, we have to manually format it
        if "glue" in dataset_name:
            for split_id in range(len(data)):
                data[split_id] = data[split_id].map(lambda x: (x['sentence'], x['label']))

        self._dataset = None
        self._train_subset = None
        self._validation_subset = None
        self._test_subset = None
        self._preprocessed = None

        if len(split) == 1:
            self._validation_type = None  # Train & evaluate on the whole dataset
            self._dataset = data[0]
        else:
            self._validation_type = 'defined_split'  # Defined by user or TFDS
            for i, s in enumerate(split):
                if s == 'train':
                    self._train_subset = data[i]
                elif s == 'validation':
                    self._validation_subset = data[i]
                elif s == 'test':
                    self._test_subset = data[i]
                self._dataset = data[i] if self._dataset is None else self._dataset.concatenate(data[i])

    @property
    def class_names(self):
        if "label" in self._info.features.keys():
            return self._info.features["label"].names
        else:
            return []

    @property
    def info(self):
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        return self._dataset

    def preprocess(self, batch_size):
        """
            Batch the dataset

            Args:
                batch_size (int): desired batch size

            Raises:
                TypeError if the batch_size is not a positive integer
                ValueError if the dataset is not defined or has already been processed
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be a positive integer")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        # Get the non-None splits
        split_list = ['_dataset', '_train_subset', '_validation_subset', '_test_subset']
        subsets = [s for s in split_list if getattr(self, s, None)]
        for subset in subsets:
            setattr(self, subset, getattr(self, subset).cache())
            setattr(self, subset, getattr(self, subset).batch(batch_size))
            setattr(self, subset, getattr(self, subset).prefetch(tf.data.AUTOTUNE))
        self._preprocessed = {'batch_size': batch_size}
