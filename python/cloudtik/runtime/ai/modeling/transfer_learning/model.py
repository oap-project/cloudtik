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


class PretrainedModel(abc.ABC):
    """
    Abstract base class for a pretrained model that can be used for transfer learning
    """

    def __init__(self, model_name: str):
        """
        Class constructor
        """
        self._model_name = model_name
        self._learning_rate = 0.001

    @property
    def model_name(self):
        """
        Name of the model
        """
        return self._model_name

    @property
    def learning_rate(self):
        """
        Learning rate for the model
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    @property
    @abc.abstractmethod
    def framework(self):
        """
        Return the framework used by the model
        """
        pass

    @abc.abstractmethod
    def load_from_directory(self, model_dir: str):
        """
        Load a model from a directory
        """
        pass

    @abc.abstractmethod
    def train(self, dataset: Dataset, output_dir, epochs=1, initial_checkpoints=None, do_eval=True):
        """
        Train the model using the specified dataset
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset: Dataset):
        """
        Evaluate the model using the specified dataset.

        Returns the loss and metrics for the model in test mode.
        """
        pass

    @abc.abstractmethod
    def predict(self, input_samples):
        """
        Generates predictions for the input samples.

        The input samples can be a BaseDataset type of object or a numpy array.
        Returns a numpy array of predictions.
        """
        pass

    @abc.abstractmethod
    def export(self, output_dir: str):
        """
        Export the serialized model to an output directory
        """
        pass

    def export_neural_compressor_config(self, config_file_path, dataset, batch_size, overwrite=False, **kwargs):
        """
        Writes a Neural Compressor compatible config file to the specified path usings args from the
        specified dataset and parameters. This is currently only supported for TF custom image classification
        datasets.

        Args:
            config_file_path (str): Destination path on where to write the .yaml config file.
            dataset (Dataset): A dataset object
            batch_size (int): Batch size to use for quantization and evaluation
            overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                              (default: False)

        Returns:
            None

        Raises:
            FileExistsError if the config file already exists and overwrite is set to False
            ValueError if the parameters are not within the expected values
            NotImplementedError if the model or dataset does not support Neural Compressor yet
        """
        raise NotImplementedError("Writing Neural Compressor config files has not be implemented for this model.")

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        """
        Performs post training quantization using the Neural Compressor on the model from the saved_model_dir
        using the specified config file. The quantized model is written to the output directory.

        Args:
            saved_model_dir (str): Source directory for the model to quantize.
            output_dir (str): Writable output directory to save the quantized model
            inc_config_path (str): Path to a Neural Compressor config file (.yaml)

        Returns:
            None

        Raises:
            NotImplementedError if the model does not support Neural Compressor yet
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            FileExistsError if the output_dir already has a saved_model.pb file
        """
        raise NotImplementedError("Post training quantization has not been implemented for this model.")

    def optimize_network(self, saved_model_dir, output_dir):
        """
        Performs FP32 network optimization on the model in the saved_model_dir
        and writes the inference-optimized model to the output_dir. Graph optimization includes converting
        variables to constants, removing training-only operations like checkpoint saving, stripping out parts
        of the graph that are never reached, removing debug operations like CheckNumerics, folding batch
        normalization ops into the pre-calculated weights, and fusing common operations into unified versions.

        Args:
            saved_model_dir (str): Source directory for the model to optimize
            output_dir (str): Writable output directory to save the optimized model

        Returns:
            None

        Raises:
            NotImplementedError if the model does not support network optimization yet
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir
            FileExistsError if the output_dir already has a saved_model.pb file
        """
        raise NotImplementedError("Network optimization has not been implemented for this model.")

    def benchmark_neural_compressor(self, saved_model_dir, inc_config_path, mode='performance'):
        """
        Use neural compressor to benchmark the specified model for performance or accuracy.

        Args:
            saved_model_dir (str): Path to the directory where the saved model is located
            inc_config_path (str): Path to neural compressor config file (.yaml)
            mode (str): performance or accuracy (defaults to performance)

        Returns:
            None

        Raises:
            NotImplementedError if the model does not support neural compressor yet
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            ValueError if an unexpected mode is provided
        """
        raise NotImplementedError("Neural compressor benchmarking is not supported for this model.")

