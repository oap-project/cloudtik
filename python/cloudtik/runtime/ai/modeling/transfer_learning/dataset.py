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

import abc


class Dataset(abc.ABC):
    """
    Abstract base class for a dataset used for training and evaluation
    """
    def __init__(self, dataset_dir, dataset_name=None):
        """
        Class constructor
        """
        self._dataset_dir = dataset_dir
        self._dataset_name = dataset_name

    @property
    def dataset_name(self):
        """
        Name of the dataset
        """
        return self._dataset_name

    @property
    def dataset_dir(self):
        """
        Host directory containing the dataset files
        """
        return self._dataset_dir

    @property
    @abc.abstractmethod
    def dataset(self):
        """
        The framework dataset object
        """
        pass

    @property
    @abc.abstractmethod
    def train_subset(self):
        """
        A subset of the dataset used for training
        """
        pass

    @property
    @abc.abstractmethod
    def validation_subset(self):
        """
        A subset of the dataset used for validation/evaluation
        """
        pass

    @property
    @abc.abstractmethod
    def test_subset(self):
        """
        A subset of the dataset held out for final testing/evaluation
        """
        pass

    @abc.abstractmethod
    def get_batch(self):
        """
        Get a single batch of images and labels from the dataset
        """
        pass
