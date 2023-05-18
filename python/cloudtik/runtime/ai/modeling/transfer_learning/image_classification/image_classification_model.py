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


class ImageClassificationModel(PretrainedModel):
    """
    Base class to represent a pretrained model for image classification
    """

    def __init__(self, image_size, do_fine_tuning: bool, dropout_layer_rate: int,
                 model_name: str):
        """
        Class constructor
        """
        self._image_size = image_size
        self._do_fine_tuning = do_fine_tuning
        self._dropout_layer_rate = dropout_layer_rate

        PretrainedModel.__init__(self, model_name)

    @property
    def image_size(self):
        """
        The fixed image size that the pretrained model expects as input, in pixels with equal width and height
        """
        return self._image_size

    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        pass

    @property
    def do_fine_tuning(self):
        """
        When True, the weights in all of the model's layers will be trainable. When False, the intermediate
        layer weights will be frozen, and only the final classification layer will be trainable.
        """
        return self._do_fine_tuning

    @property
    def dropout_layer_rate(self):
        """
        The probability of any one node being dropped when a dropout layer is used
        """
        return self._dropout_layer_rate

    def get_inc_config_template_dict(self):
        """
        Returns a dictionary for a config template compatible with the Intel Neural Compressor.

        It loads the yaml file image_classification_template.yaml and then fills in parameters
        that the model knows about (like framework and model name). There are still more parameters that need to be
        filled in before using the config with INC (like the dataset information, image size, etc).
        """
        this_dir = os.path.dirname(__file__)
        template_file_path = os.path.join(this_dir, "image_classification_template.yaml")

        if not os.path.exists(template_file_path):
            raise FileNotFoundError("Unable to find the image recognition config template at:", template_file_path)

        with open(template_file_path, 'r') as template_yaml:
            config_template = yaml.safe_load(template_yaml)

        # Update parameters that we know in the template
        config_template["model"]["framework"] = str(self.framework)
        config_template["model"]["name"] = self.model_name

        return config_template
