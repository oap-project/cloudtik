#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
from pydoc import locate

from cloudtik.runtime.ai.modeling.transfer_learning import BASE_DIR
from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.sources import ModelSource
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import read_json_file


class ModelDownloader:
    """
    A unified model downloader class.

    Can download models from TF Hub, Torchvision, and Hugging Face.
    """
    def __init__(self, model_name, source, model_dir=None, **kwargs):
        """
        Class constructor for a ModelDownloader.

            Args:
                model_name (str): Name of the model
                source (str, optional): The source to download the model from; options are 'tf_hub',
                    'torchvision', 'pytorch_hub', 'hugging_face', and 'keras'
                model_dir (str): Local destination directory of the model, if None the model hub's default cache
                    directory will be used
                kwargs (optional): Some model hubs accept additional keyword arguments when downloading

        """
        if model_dir is not None and not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        self._model_name = model_name
        self._model_dir = model_dir
        self._source = ModelSource.from_str(source)
        self._args = kwargs

    def download(self):
        """
        Download the model

            Returns:
                A torch.nn.Module, keras.engine.functional.Functional, or tensorflow_hub.keras_layer.KerasLayer object

        """
        if self._source == ModelSource.TF_HUB:
            from tensorflow_hub import KerasLayer
            if self._model_dir is not None:
                os.environ['TFHUB_CACHE_DIR'] = self._model_dir

            return KerasLayer(self._model_name, **self._args)

        elif self._source == ModelSource.TORCHVISION:
            if self._model_dir is not None:
                os.environ['TORCH_HOME'] = self._model_dir
            pretrained_model_class = locate('torchvision.models.{}'.format(self._model_name))

            return pretrained_model_class(**self._args)

        elif self._source == ModelSource.PYTORCH_HUB:
            import torch

            if self._model_dir is not None:
                os.environ['TORCH_HOME'] = self._model_dir

            config_file = os.path.join(
                BASE_DIR,
                "image_classification/pytorch/torchvision/pytorch_hub/image_classification_models.json")
            pytorch_hub_model_map = read_json_file(config_file)
            self._repo = pytorch_hub_model_map[self._model_name]["repo"]

            # Some models have pretrained=True by default, which error out if passed in load()
            if pytorch_hub_model_map[self._model_name]["pretrained_default"] == "True":
                return torch.hub.load(self._repo, self._model_name)
            else:
                return torch.hub.load(self._repo, self._model_name, pretrained=True)

        elif self._source == ModelSource.HUGGING_FACE:
            if self._model_dir is not None:
                os.environ['TRANSFORMERS_CACHE'] = self._model_dir
            # AutoModelForSequenceClassification is currently the only supported model type
            from transformers import AutoModelForSequenceClassification

            return AutoModelForSequenceClassification.from_pretrained(self._model_name, **self._args)

        elif self._source == ModelSource.KERAS_APPLICATIONS:
            if self._model_dir is not None:
                os.environ['KERAS_HOME'] = self._model_dir
            try:
                pretrained_model_class = locate('keras.applications.{}'.format(self._model_name))
            except TypeError:
                pretrained_model_class = locate('keras.applications.{}.{}'.format(self._model_name.lower(),
                                                                                  self._model_name))

            return pretrained_model_class(**self._args)
