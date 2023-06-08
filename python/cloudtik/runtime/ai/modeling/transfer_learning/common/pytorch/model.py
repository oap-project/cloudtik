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
import datetime
import inspect
import os
import shutil

import dill
import numpy
import random
import torch

from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch import get_train_script
from cloudtik.runtime.ai.modeling.transfer_learning.model import PretrainedModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import verify_directory
from cloudtik.runtime.ai.runner import run_command


class PyTorchModel(PretrainedModel):
    """
    Base class to represent a PyTorch model
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._model = None  # This gets initialized later
        self._optimizer = None  # This gets initialized later
        self._optimizer_class = None  # This gets initialized in subclass
        self._loss = None  # This gets initialized in subclass
        self._lr_scheduler = None
        self._history = {}

        # Setup warnings module to set warnings to go to stdout
        import warnings
        import sys

        def customwarn(message, category, filename, lineno, file=None, line=None):
            sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
        warnings.showwarning = customwarn

    @property
    def framework(self):
        return "pytorch"

    def _set_seed(self, seed):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

    def _check_train_inputs(self, output_dir, dataset, dataset_type, epochs, initial_checkpoints):
        verify_directory(output_dir)

        if not isinstance(dataset, dataset_type):
            raise TypeError("The dataset must be a {} but found a {}".format(dataset_type, type(dataset)))

        if not dataset.info['preprocessing_info']:
            raise ValueError("Dataset hasn't been preprocessed yet.")

        if not isinstance(epochs, int):
            raise TypeError("Invalid type for the epochs arg. Expected an int but found a {}".format(type(epochs)))

        if initial_checkpoints and not isinstance(initial_checkpoints, str):
            raise TypeError("The initial_checkpoints parameter must be a string but found a {}".format(
                type(initial_checkpoints)))

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

    def _update_history(self, key, value):
        if key not in self._history:
            self._history[key] = []
        self._history[key].extend([value])

    def load_from_directory(self, model_dir: str):
        """
        Load a saved model from the model_dir path
        """
        # Verify that the model directory exists
        verify_directory(model_dir, require_directory_exists=True)
        model_copy = torch.load(os.path.join(model_dir, 'model.pt'))
        self._model = dill.loads(model_copy)
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

    def list_layers(self, verbose=False):
        """
        Lists all of the named modules (e.g. features, avgpool, classifier) and layers
        (ReLU, MaxPool2d, Dropout, Linear, etc) in a given PyTorch model

        Args:
            verbose (bool): True/False option set by default to be False, displays only high-level modules
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be summarized.')

        # Display a high-level list of the modules e.g. features, avgpool, classifier
        print("\nModel Layers\n============")
        for (name, module) in self._model.named_children():
            if not verbose or not list(module.named_children()):
                print('{}: {}/{} parameters are trainable'.format(
                    name, sum(p.numel() for p in module.parameters() if p.requires_grad),
                    sum(p.numel() for p in module.parameters())))
            else:
                print('{}:'.format(name))
                for (layer_name, layer) in module.named_children():
                    print('  {}: {}/{} parameters are trainable'.format(
                        layer_name, sum(p.numel() for p in layer.parameters() if p.requires_grad),
                        sum(p.numel() for p in layer.parameters())))

        trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print('\nTotal Trainable Parameters: {}/{}'.format(
            trainable_parameters,
            sum(p.numel() for p in self._model.parameters())))

        return trainable_parameters

    def freeze_layer(self, layer_name):
        """
        Freezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be frozen.')

        # Freeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_layer(self, layer_name):
        """
        Unfreezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be unfrozen.')

        # Unfreeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = True

    def save_objects(self, dataset, output_dir, temp_dir):
        """
        Helper function to export dataset and model objects to disk for distributed job

        Args:
            dataset: Dataset object to save. It must be an object of
                Dataset with info, train, test, and validation
                subsets properties which can be accessed.
            output_dir (str): Path to a directory where the dataset and model objects are saved.
                The path/file of the save objects is returned.
            temp_dir (str): The temp data dir at local.
        """
        now = datetime.datetime.now()
        filename = f"torch_objects_{now:%Y-%m-%d_%H-%M-%S}.obj"
        objects_path = os.path.join(output_dir, filename)

        # save to temporary path and then move
        if temp_dir:
            save_objects_path = os.path.join(temp_dir, filename)
        else:
            save_objects_path = objects_path

        objects_to_save = {
            "dataset": dataset.dataset,
            "info": dataset.info,
            "train_subset": dataset.train_subset,
            "test_subset": dataset.test_subset,
            "validation_subset": dataset.validation_subset,
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": self._loss
        }

        torch.save(objects_to_save, save_objects_path)

        if temp_dir:
            shutil.move(save_objects_path, objects_path)

        return objects_path

    @staticmethod
    def fit_distributed(
            nnodes, nproc_per_node, hosts, hostfile,
            epochs, batch_size, ipex_optimize,
            objects_path, category):
        train_script = get_train_script()
        command = [
            train_script, "--objects-path", objects_path,
            "--category", category,
            "--epochs", str(epochs), "--batch-size", str(batch_size)
        ]
        if ipex_optimize:
            command += ['--ipex']
            command += ['--backend', 'ccl']

        run_command(
            command,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            hosts=hosts,
            hostfile=hostfile
        )
