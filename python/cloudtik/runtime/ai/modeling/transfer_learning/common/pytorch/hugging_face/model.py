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

from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.model import PyTorchModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import verify_directory


class HuggingFaceModel(PyTorchModel):
    """
    Base class to represent a Hugging Face model
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, epochs, extra_layers):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not dataset.info['preprocessing_info']:
            raise ValueError("Dataset hasn't been preprocessed yet.")

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if extra_layers:
            if not isinstance(extra_layers, list) or not all(isinstance(n, int) for n in extra_layers):
                raise ValueError("extra_layers argument must be a list of integers but found a {}".format(extra_layers))
