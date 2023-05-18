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
import os
import yaml

from cloudtik.runtime.ai.modeling.transfer_learning.model import PretrainedModel


class TextClassificationModel(PretrainedModel):
    """
    Class to represent a pretrained model for text classification
    """

    def __init__(self, model_name: str, dropout_layer_rate: float):
        self._dropout_layer_rate = dropout_layer_rate
        PretrainedModel.__init__(self, model_name)

        # Default learning rate for text models
        self._learning_rate = 3e-5

    def get_inc_config_template_dict(self):
        """
        Returns a dictionary for a config template compatible with the Intel Neural Compressor.

        It loads the yaml file tlt/models/configs/inc/text_classification_template.yaml and then fills in parameters
        that the model knows about (like framework and model name). There are still more parameters that need to be
        filled in before using the config with INC (like the dataset information, size, etc).
        """
        this_dir = os.path.dirname(__file__)
        template_file_path = os.path.join(this_dir, "text_classification_template.yaml")

        if not os.path.exists(template_file_path):
            raise FileNotFoundError("Unable to find the config template at:", template_file_path)

        with open(template_file_path, 'r') as template_yaml:
            config_template = yaml.safe_load(template_yaml)

        # Update parameters that we know in the template
        config_template["model"]["framework"] = str(self.framework)
        config_template["model"]["name"] = self.model_name65

        return config_template

    @property
    @abc.abstractmethod
    def num_classes(self):
        pass

    @property
    def dropout_layer_rate(self):
        return self._dropout_layer_rate

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False, **kwargs):
        raise NotImplementedError("Writing INC config files has not be implemented yet for text classification")

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        raise NotImplementedError("Post training quantization has not been implemented yet for text classification")

    def optimize_graph(self, saved_model_dir, output_dir):
        raise NotImplementedError("Optimize graph has not been implemented yet for text classification")

    def benchmark(self, saved_model_dir, inc_config_path, mode='performance'):
        raise NotImplementedError("Benchmarking has not been implemented yet for text classification")
