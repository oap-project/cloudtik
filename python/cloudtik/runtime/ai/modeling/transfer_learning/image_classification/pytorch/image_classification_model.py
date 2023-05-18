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

import copy
import inspect
import os
import time
import dill
import yaml
import subprocess

from tqdm import tqdm
import torch

from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import \
    verify_directory, validate_model_name
from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.model import PyTorchModel
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_dataset import \
    ImageClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_model import \
    ImageClassificationModel
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.pytorch.image_classification_dataset import \
    PyTorchImageClassificationDataset


class PyTorchImageClassificationModel(ImageClassificationModel, PyTorchModel):
    """
    Class to represent a PyTorch model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        # PyTorch models generally do not enforce a fixed input shape
        self._image_size = 'variable'

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._lr_scheduler = None
        self._generate_checkpoints = True

        # placeholder for model definition
        self._model = None
        self._num_classes = None

        PyTorchModel.__init__(self, model_name)
        ImageClassificationModel.__init__(self, self._image_size, self._do_fine_tuning, self._dropout_layer_rate,
                                          self._model_name)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        self._optimizer_class = optimizer if optimizer else torch.optim.Adam
        self._opt_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else torch.nn.CrossEntropyLoss
        self._loss_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        if model is None:
            self._model = None
        elif isinstance(model, str):
            self.load_from_directory(model)
            layers = list(self._model.children())
            if isinstance(layers[-1], torch.nn.Sequential):
                self._num_classes = layers[-1][-1].out_features
            else:
                self._num_classes = layers[-1].out_features
        elif isinstance(model, torch.nn.Module):
            self._model = model
            layers = list(self._model.children())
            if isinstance(layers[-1], torch.nn.Sequential):
                self._num_classes = layers[-1][-1].out_features
            else:
                self._num_classes = layers[-1].out_features
        else:
            raise TypeError("The model input must be a torch.nn.Module, string or",
                            "None but found a {}". format(type(model)))

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    @property
    def framework(self):
        return "pytorch"

    def _fit(self, output_dir, dataset, epochs, do_eval, early_stopping, lr_decay):
        """Main PyTorch training loop"""
        since = time.time()

        device = torch.device(self._device)
        self._model = self._model.to(device)

        if dataset.train_subset:
            train_data_loader = dataset.train_loader
            data_length = len(dataset.train_subset)
        else:
            train_data_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        if do_eval and dataset.validation_subset:
            validation_data_loader = dataset.validation_loader
            validation_data_length = len(dataset.validation_subset)
        else:
            validation_data_loader = None
            validation_data_length = 0

        # For early stopping, if enabled
        patience = 10
        trigger_time = 0
        last_loss = 1.0

        if lr_decay:
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=0.2, patience=5,
                                                                            cooldown=1, min_lr=0.0000000001)

        self._history = {}
        self._model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            # Training phase
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Forward and backward pass
                with torch.set_grad_enabled(True):
                    outputs = self._model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self._loss(outputs, labels)
                    loss.backward()
                    self._optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_epoch_loss = running_loss / data_length
            train_epoch_acc = float(running_corrects) / data_length
            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                self._model.eval()
                running_loss = 0.0
                running_corrects = 0

                with torch.no_grad():
                    print("Performing Evaluation")
                    for inputs, labels in tqdm(validation_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self._loss(outputs, labels)

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                eval_epoch_loss = running_loss / validation_data_length
                eval_epoch_acc = float(running_corrects) / validation_data_length
                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

                if lr_decay:
                    lr = self._lr_scheduler.optimizer.param_groups[0]['lr']
                    self._update_history('LR', lr)
                    loss_acc_output += f' - LR: {lr:.4f}'
                    self._lr_scheduler.step(eval_epoch_loss)

            if early_stopping:
                if eval_epoch_loss >= last_loss:
                    trigger_time += 1

                    if trigger_time >= patience:
                        # Stop Early
                        print("Early stopping has been triggered after " + str(epoch) + " epochs.")
                        break
                else:
                    trigger_time = 0

                last_loss = eval_epoch_loss

            print(loss_acc_output)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if self._generate_checkpoints:
            valid_model_name = validate_model_name(self.model_name)
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
            verify_directory(checkpoint_dir)
            try:
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
            except KeyError:
                # Calling state_dict() on an IPEX optimizer calls into the torch optimizer's __setstate__ method
                # which in PyTorch 1.12 assumes that the first state value will always have a 'step' key
                state_values = list(self._optimizer.state.values())
                if 'step' not in state_values[0].keys():
                    state_values[0]['step'] = torch.tensor([])
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

    def _fit_distributed(self, hostfile, nnodes, nproc_per_node, epochs, batch_size, ipex_optimize):
        # TODO: for distributed
        # distributed_vision_script = os.path.join(TLT_DISTRIBUTED_DIR, "run_train_pyt.py")
        # distributed_vision_script = "run_train_pyt.py"

        default_port = '29500'
        default_master_addr = '127.0.0.1'

        addresses = []

        if hostfile is not None:
            if os.path.isfile(hostfile):
                # if addresses are given as line separated IP addresses
                with open(hostfile) as hf:
                    addresses = hf.readlines()
                addresses = [a.strip('\n') for a in addresses]
            else:
                # if addresses are given as a comma separated IP addresses
                addresses = hostfile.split(',')

            default_master_addr = addresses[0]

            # If port is given in the format of "0.0.0.0:9999"
            if ':' in default_master_addr:
                colon_index = default_master_addr.index(':')
                default_port = default_master_addr[colon_index + 1:]
                default_master_addr = default_master_addr[:colon_index]

                # We create/rewrite the hostfile to contain only IP addresses
                with open('hostfile', 'w') as hf:
                    for addr in addresses:
                        if ':' in addr:
                            addr = addr[:addr.index(':')]
                        hf.write(addr + '\n')
                hostfile = 'hostfile'

        bash_command = 'python -m intel_extension_for_pytorch.cpu.launch --distributed'
        bash_command += ' --hostfile {}'.format(hostfile)
        bash_command += ' --nnodes {}'.format(nnodes)
        bash_command += ' --nproc_per_node {}'.format(nproc_per_node)
        bash_command += ' {}'.format(distributed_vision_script)
        bash_command += ' --master_addr {}'.format(default_master_addr)
        bash_command += ' --master_port {}'.format(default_port)
        bash_command += ' --backend {}'.format('ccl')
        bash_command += ' --use_case {}'.format('image_classification')
        bash_command += ' --epochs {}'.format(epochs)
        bash_command += ' --batch_size {}'.format(batch_size)
        if not ipex_optimize:
            bash_command += ' --disable_ipex'

        print(bash_command)
        subprocess.run(bash_command.split(' '))

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, ipex_optimize=False, distributed=False,
              hostfile=None, nnodes=1, nproc_per_node=1):
        """
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the model from torchvision and add on a fully-connected dense layer with linear activation
            based on the number of classes in the specified dataset. The model and optimizer are defined and trained
            for the specified number of epochs.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                epochs (int): Number of epochs to train the model (default: 1)
                initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                early_stopping (bool): Enable early stopping if convergence is reached while training
                    at the end of each epoch.
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                seed (int): Optionally set a seed for reproducibility.
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to False.
                distributed (bool): Boolean flag to use distributed training. Defaults to False.
                hostfile (str): Name of the hostfile for distributed training. Defaults to None.
                nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
                nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
                to 1.

            Returns:
                Trained PyTorch model object
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints,
                                 distributed, hostfile)

        dataset_num_classes = len(dataset.class_names)

        # Check that the number of classes matches the model outputs
        if dataset_num_classes != self.num_classes:
            raise RuntimeError("The number of model outputs ({}) differs from the number of dataset classes ({})".
                               format(self.num_classes, dataset_num_classes))

        self._set_seed(seed)

        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if distributed:
            # TODO: for distributed
            # self.export_for_distributed(TLT_DISTRIBUTED_DIR, dataset)
            batch_size = dataset._preprocessed['batch_size']
            self._fit_distributed(hostfile, nnodes, nproc_per_node, epochs, batch_size, ipex_optimize)

        else:
            # Call ipex.optimize
            if ipex_optimize:
                import intel_extension_for_pytorch as ipex
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)
            self._fit(output_dir, dataset, epochs, do_eval, early_stopping, lr_decay)

        return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False):
        """
        Evaluate the accuracy of the model on a dataset.
        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_loader = dataset.test_loader
                data_length = len(dataset.test_subset)
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_loader = dataset.validation_loader
            data_length = len(dataset.validation_subset)
        else:
            eval_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        model = self._model
        optimizer = self._optimizer

        # Do the evaluation
        device = torch.device(self._device)
        model = model.to(device)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(eval_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self._loss(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_length
        epoch_acc = float(running_corrects) / data_length

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return [epoch_loss, epoch_acc]

    def predict(self, input_samples, return_type='class'):
        """
        Perform feed-forward inference and predict the classes of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            return_type (str): Using 'class' will return the highest scoring class (default), using 'scores' will
                               return the raw output/logits of the last layer of the network, using 'probabilities' will
                               return the output vector after applying a softmax function (so results sum to 1)

        Returns:
            List of classes, probability vectors, or raw score vectors

        Raises:
            ValueError if the return_type is not one of 'class', 'probabilities', or 'scores'
        """
        return_types = ['class', 'probabilities', 'scores']
        if not isinstance(return_type, str) or return_type not in return_types:
            raise ValueError('Invalid return_type ({}). Expected one of {}.'.format(return_type, return_types))

        self._model.eval()
        with torch.no_grad():
            predictions = self._model(input_samples)
        if return_type == 'class':
            _, predicted_ids = torch.max(predictions, 1)
            return predicted_ids
        elif return_type == 'probabilities':
            return torch.nn.functional.softmax(predictions)
        else:
            return predictions

    def export(self, output_dir):
        """
        Save a serialized version of the model to the output_dir path
        """
        if self._model:
            # Save the model in a format that can be re-loaded for inference
            verify_directory(output_dir)
            valid_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, valid_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")
            verify_directory(saved_model_dir)
            model_copy = dill.dumps(self._model)
            torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False,
                              resize_interpolation='bicubic', accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                              exit_policy_max_trials=50, tuning_random_seed=9527,
                              tuning_workspace=''):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters.

        Args:
            config_file_path (str): Destination path on where to write the .yaml config file.
            dataset (BaseDataset): A tlt dataset object
            batch_size (int): Batch size to use for quantization and evaluation
            overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                              (default: False)
            resize_interpolation (str): Interpolation type. Select from: 'bilinear', 'nearest', 'bicubic'
                                        (default: bicubic)
            accuracy_criterion_relative (float): Relative accuracy loss (default: 0.01, which is 1%)
            exit_policy_timeout (int): Tuning timeout in seconds (default: 0). Tuning processing finishes when the
                                       timeout or max_trials is reached. A tuning timeout of 0 means that the tuning
                                       phase stops when the accuracy criterion is met.
            exit_policy_max_trials (int): Maximum number of tuning trials (default: 50). Tuning processing finishes when
                                          the timeout or or max_trials is reached.
            tuning_random_seed (int): Random seed for deterministic tuning (default: 9527).
            tuning_workspace (dir): Path the INC nc_workspace folder. If the string is empty and the OUTPUT_DIR env var
                                    is set, that output directory will be used. If the string is empty and the
                                    OUTPUT_DIR env var is not set, the default INC nc_workspace location will be used.
        Returns:
            None
        Raises:
            FileExistsError if the config file already exists and overwrite is set to False.
            ValueError if the parameters are not within the expected values
            NotImplementedError if the dataset type is not TFCustomImageClassificationDataset.
        """
        if os.path.isfile(config_file_path) and not overwrite:
            raise FileExistsError('A file already exists at: {}. Provide a new file path or set overwrite=True',
                                  config_file_path)

        # We can setup the a custom dataset to use the ImageFolder dataset option in INC.
        # They don't have a PyTorch Dataset option, so for now, we only support custom datasets for quantization
        if dataset is not PyTorchImageClassificationDataset \
                and type(dataset) != PyTorchImageClassificationDataset:
            raise NotImplementedError('quantization has only been implemented for PyTorch image classification models '
                                      'with custom datasets')

        if batch_size and not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Invalid value for batch size ({}). Expected a positive integer.'.format(batch_size))

        if resize_interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Invalid value for resize interpolation ({}). Expected one of the following values: '
                             'bilinear, nearest, bicubic'.format(resize_interpolation))

        if accuracy_criterion_relative and not isinstance(accuracy_criterion_relative, float) or \
                not (0.0 <= accuracy_criterion_relative <= 1.0):
            raise ValueError('Invalid value for the accuracy criterion ({}). Expected a float value between 0.0 '
                             'and 1.0'.format(accuracy_criterion_relative))

        if exit_policy_timeout and not isinstance(exit_policy_timeout, int) or exit_policy_timeout < 0:
            raise ValueError('Invalid value for the exit policy timeout ({}). Expected a positive integer or 0.'.
                             format(exit_policy_timeout))

        if exit_policy_max_trials and not isinstance(exit_policy_max_trials, int) or exit_policy_max_trials < 1:
            raise ValueError('Invalid value for max trials ({}). Expected an integer greater than 0.'.
                             format(exit_policy_timeout))

        if tuning_random_seed and not isinstance(tuning_random_seed, int) or tuning_random_seed < 0:
            raise ValueError('Invalid value for tuning random seed ({}). Expected a positive integer.'.
                             format(tuning_random_seed))

        if not isinstance(tuning_workspace, str):
            raise ValueError('Invalid value for the nc_workspace directory. Expected a string.')

        # Get the image recognition Intel Neural Compressor template
        config_template = ImageClassificationModel.get_inc_config_template_dict(self)

        # Collect the different data loaders into a list, so that we can update them all the with the data transforms
        dataloader_configs = []

        # If tuning_workspace is undefined, use the OUTPUT_DIR, if the env var exists
        if not tuning_workspace:
            output_dir_env_var = os.getenv('OUTPUT_DIR', '')

            if output_dir_env_var:
                tuning_workspace = os.path.join(output_dir_env_var, 'nc_workspace')

        print("tuning_workspace:", tuning_workspace)

        if "quantization" in config_template.keys() and "calibration" in config_template["quantization"].keys() \
                and "dataloader" in config_template["quantization"]["calibration"].keys():
            dataloader_configs.append(config_template["quantization"]["calibration"]["dataloader"])

        if "evaluation" in config_template.keys():
            if "accuracy" in config_template["evaluation"].keys() and \
                    "dataloader" in config_template["evaluation"]["accuracy"].keys():
                dataloader_configs.append(config_template["evaluation"]["accuracy"]["dataloader"])
            if "performance" in config_template["evaluation"].keys() and \
                    "dataloader" in config_template["evaluation"]["performance"].keys():
                dataloader_configs.append(config_template["evaluation"]["performance"]["dataloader"])

        transform_config = {
            "Resize": {
                "size": self._image_size
            },
            "CenterCrop": {
                "size": self._image_size
            },
            "ToTensor": {},
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }

        del config_template["evaluation"]["accuracy"]["postprocess"]

        config_template["quantization"]["approach"] = "post_training_dynamic_quant"

        # Update the data loader configs
        for dataloader_config in dataloader_configs:
            # Set the transform configs for resizing and rescaling
            dataloader_config["transform"] = copy.deepcopy(transform_config)

            # Update dataset directory for the custom dataset
            if "dataset" in dataloader_config.keys() and "ImageFolder" in dataloader_config["dataset"].keys():
                dataloader_config["dataset"]["ImageFolder"]["root"] = dataset.dataset_dir

            dataloader_config["batch_size"] = batch_size

        if "tuning" in config_template.keys():
            config_template["tuning"]["accuracy_criterion"]["relative"] = accuracy_criterion_relative

            if exit_policy_timeout is None:
                config_template["tuning"]["exit_policy"].pop('timeout', None)
            else:
                config_template["tuning"]["exit_policy"]["timeout"] = exit_policy_timeout

            if exit_policy_max_trials is None:
                config_template["tuning"]["exit_policy"].pop('max_trials', None)
            else:
                config_template["tuning"]["exit_policy"]["max_trials"] = exit_policy_max_trials

            if tuning_random_seed is None:
                config_template["tuning"].pop('random_seed', None)
            else:
                config_template["tuning"]["random_seed"] = tuning_random_seed

            if tuning_workspace:
                if "workspace" not in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"] = {}

                config_template["tuning"]["workspace"]["path"] = tuning_workspace
            else:
                # No tuning_workspace is defined, so remove it from the config
                if "workspace" in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"].pop("path", None)

                    if len(config_template["tuning"]["workspace"].keys()) == 0:
                        config_template["tuning"].pop("workspace", None)

        # Create the directory where the file will be written, if it doesn't already exist
        if not os.path.exists(os.path.dirname(config_file_path)):
            os.makedirs(os.path.dirname(config_file_path))

        # Write the config file
        with open(config_file_path, "w") as config_file:
            yaml.dump(config_template, config_file, sort_keys=False)

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        """
        Performs post training quantization using the Intel Neural Compressor on the model from the saved_model_dir
        using the specified config file. The quantized model is written to the output directory.

        Args:
            saved_model_dir (str): Source directory for the model to quantize.
            output_dir (str): Writable output directory to save the quantized model
            inc_config_path (str): Path to an INC config file (.yaml)

        Returns:
            None

        Raises:
            NotADirectoryError if the model is not a directory
            FileNotFoundError if a model.pt is not found in the model or if the inc_config_path file
            is not found.
            FileExistsError if the output_dir already has a model.pt file
        """
        # The saved model directory should exist and contain a model.pt file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "model.pt")):
            raise FileNotFoundError("The saved model directory ({}) should have a model.pt file".format(
                saved_model_dir))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "model.pt")):
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "model.pt"))

        from neural_compressor.experimental import Quantization
        quantizer = Quantization(inc_config_path)
        quantizer.model = self._model
        quantized_model = quantizer.fit()

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)
            import subprocess
            # Change the model filename from best_model.pt to model.pt to match our convention
            p = subprocess.Popen(["mv", output_dir + "/best_model.pt", output_dir + "/model.pt"],
                                 stdout=subprocess.PIPE)
            stdout, stderr = p.communicate()

    def benchmark(self, saved_model_dir, inc_config_path, mode='performance', model_type='fp32'):
        """
        Use INC to benchmark the specified model for performance or accuracy. You must specify whether the
        input model is fp32 or int8. IPEX int8 models are not supported yet.

        Args:
            saved_model_dir (str): Path to the directory where the saved model is located
            inc_config_path (str): Path to an INC config file (.yaml)
            mode (str): Performance or accuracy (defaults to performance)
            model_type (str): Floating point (fp32) or quantized integer (int8) model type
        Returns:
            None
        Raises:
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a model.pt is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            ValueError if an unexpected mode is provided
        """
        # The saved model directory should exist and contain a model.pt file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "model.pt")):
            raise FileNotFoundError("The saved model directory ({}) should have a model.pt file".format(
                saved_model_dir))

        # Validate mode
        if mode not in ['performance', 'accuracy']:
            raise ValueError("Invalid mode: {}. Expected mode to be 'performance' or 'accuracy'.".format(mode))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        from neural_compressor.experimental import Benchmark, common
        if model_type == "fp32":
            evaluator = Benchmark(inc_config_path)
            evaluator.model = self._model
            return evaluator(mode)
        elif model_type == "int8":
            try:
                from neural_compressor.utils.pytorch import load
                evaluator = Benchmark(inc_config_path)
                evaluator.model = common.Model(load(os.path.join(saved_model_dir, 'model.pt'), self._model))
                return evaluator(mode)
            except AssertionError:
                raise NotImplementedError("This model type is not yet supported by INC benchmarking")

    def export_for_distributed(self, output_dir, dataset):
        """
        Helper function to export dataset and model objects to disk for distributed job

        Args:
            output_dir (str): Path to a directory where the dataset and model objects are saved.
                Default file name for saving the objects is "torch_saved_objects.obj"
            dataset (ImageClassificationDataset): Dataset object to save. It must be an object of
                ImageClassificationDataset so that the dataset info, train, test, and validation
                subsets can be accessed.
        """

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
        torch.save(objects_to_save, os.path.join(output_dir, "torch_saved_objects.obj"))
