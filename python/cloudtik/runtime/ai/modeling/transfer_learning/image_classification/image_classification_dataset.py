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

from ..common.dataset import Dataset


class ImageClassificationDataset(Dataset):
    """
    Base class for an image classification dataset
    """
    def __init__(self, dataset_dir, dataset_name=""):
        """
        Class constructor
        """
        Dataset.__init__(self, dataset_dir, dataset_name)

    @property
    @abc.abstractmethod
    def class_names(self):
        """
        Returns the list of class names (abstract method implemented by subclasses)
        """
        pass
