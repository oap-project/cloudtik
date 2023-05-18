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

from cloudtik.runtime.ai.modeling.transfer_learning.dataset import Dataset


class TextClassificationDataset(Dataset):
    """
    Base class for a text classification dataset
    """
    def __init__(self, dataset_dir, dataset_name=""):
        Dataset.__init__(self, dataset_dir, dataset_name)

    @property
    @abc.abstractmethod
    def class_names(self):
        pass

    def get_str_label(self, numerical_value):
        """
            Returns the string label (class name) associated with the specified numerical value. If the numerical
            value provided is a float, it will be rounded to the nearest integer.

            Args:
                numerical_value (int or float): Numerical label value

            Raises:
                TypeError if the numerical value is not a float or an integer
                ValueError if the numerical value does not map to a class label
        """
        if isinstance(numerical_value, float):
            numerical_value = int(round(numerical_value))

        if not isinstance(numerical_value, int):
            raise TypeError("Invalid type for the numerical value. Expected an integer or float value.")

        if len(self.class_names) > numerical_value:
            return self.class_names[numerical_value]
        else:
            raise ValueError("The numerical value {} exceeds the number of classes in the dataset ({})".format(
                numerical_value, len(self.class_names)))
