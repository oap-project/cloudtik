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
import inspect
import torch

from cloudtik.runtime.ai.modeling.transfer_learning.model import PretrainedModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import verify_directory


class HuggingFaceModel(PretrainedModel):
    """
    Base class to represent a Hugging Face model
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._history = {}

    def _update_history(self, key, value):
        if key not in self._history:
            self._history[key] = []
        self._history[key].extend([value])

    def _check_optimizer_loss(self, optimizer, loss):
        if optimizer is not None and (not inspect.isclass(optimizer) or
                                      torch.optim.Optimizer not in inspect.getmro(optimizer)):
            raise TypeError("The optimizer input must be a class (not an instance) of type torch.optim.Optimizer or "
                            "None but found a {}. Example: torch.optim.AdamW".format(type(optimizer)))
        if loss is not None and (not inspect.isclass(loss) or
                                 torch.nn.modules.loss._Loss not in inspect.getmro(loss)):
            raise TypeError("The optimizer input must be a class (not an instance) of type "
                            "torch.nn.modules.loss._Loss or None but found a {}. "
                            "Example: torch.nn.CrossEntropyLoss".format(type(loss)))

    def _check_train_inputs(self, output_dir, dataset, dataset_type, extra_layers, epochs):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not dataset.info['preprocessing_info']:
            raise ValueError("Dataset hasn't been preprocessed yet.")

        if extra_layers:
            if not isinstance(extra_layers, list) or not all(isinstance(n, int) for n in extra_layers):
                raise ValueError("extra_layers argument must be a list of integers but found a {}".format(extra_layers))

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))
