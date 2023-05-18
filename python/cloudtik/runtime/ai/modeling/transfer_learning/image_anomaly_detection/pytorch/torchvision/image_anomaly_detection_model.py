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

from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.models import ModelDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import read_json_file
from cloudtik.runtime.ai.modeling.transfer_learning.image_anomaly_detection.pytorch.image_anomaly_detection_model \
    import PyTorchImageAnomalyDetectionModel


class TorchvisionImageAnomalyDetectionModel(PyTorchImageAnomalyDetectionModel):
    """
    Class to represent a Torchvision pretrained model for anomaly detection
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        PyTorchImageAnomalyDetectionModel.__init__(self, model_name, **kwargs)

        this_dir = os.path.dirname(__file__)
        torchvision_model_map = read_json_file(os.path.join(
            this_dir, "image_anomaly_detection_models.json"))
        if model_name not in torchvision_model_map.keys():
            raise ValueError("The specified Torchvision image anomaly detection model ({}) "
                             "is not supported.".format(model_name))

        self._image_size = torchvision_model_map[model_name]["image_size"]
        self._original_dataset = torchvision_model_map[model_name]["original_dataset"]

        downloader = ModelDownloader(
            model_name, source='torchvision',
            model_dir=None, weights=self._original_dataset)
        self._model = downloader.download()
