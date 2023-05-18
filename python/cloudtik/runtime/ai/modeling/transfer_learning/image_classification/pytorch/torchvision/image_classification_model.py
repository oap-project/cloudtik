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
from tqdm import tqdm
import torch

from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_dataset \
    import ImageClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.models \
    import ModelDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import read_json_file
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.pytorch.image_classification_model \
    import PyTorchImageClassificationModel


class TorchvisionImageClassificationModel(PyTorchImageClassificationModel):
    """
    Class to represent a Torchvision pretrained model for image classification
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        this_dir = os.path.dirname(__file__)
        model_map = read_json_file(os.path.join(
            this_dir, "image_classification_models.json"))
        if model_name not in model_map.keys():
            raise ValueError("The specified Torchvision image classification model ({}) "
                             "is not supported.".format(model_name))

        PyTorchImageClassificationModel.__init__(self, model_name, **kwargs)

        self._classification_layer = model_map[model_name]["classification_layer"]
        self._image_size = model_map[model_name]["image_size"]
        self._original_dataset = model_map[model_name]["original_dataset"]

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._distributed = False

    def _model_downloader(self, model_name):
        downloader = ModelDownloader(model_name, source='torchvision',
                                     model_dir=None, weights=self._original_dataset)
        model = downloader.download()
        return model

    def _get_hub_model(self, num_classes, ipex_optimize=False, extra_layers=None):
        if not self._model:
            self._model = self._model_downloader(self._model_name)

            if not self._do_fine_tuning:
                for param in self._model.parameters():
                    param.requires_grad = False

            # Do not apply a softmax activation to the final layer as with TF because loss can be affected
            if len(self._classification_layer) == 2:
                base_model = getattr(self._model, self._classification_layer[0])
                classifier = getattr(self._model, self._classification_layer[0])[self._classification_layer[1]]
                self._model.classifier = base_model[0: self._classification_layer[1]]
                num_features = classifier.in_features
                if extra_layers:
                    for layer in extra_layers:
                        self._model.classifier.append(torch.nn.Linear(num_features, layer))
                        self._model.classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                self._model.classifier.append(torch.nn.Linear(num_features, num_classes))
            else:
                classifier = getattr(self._model, self._classification_layer[0])
                if self._classification_layer[0] == "heads":
                    num_features = classifier.head.in_features
                else:
                    num_features = classifier.in_features

                if extra_layers:
                    setattr(self._model, self._classification_layer[0], torch.nn.Sequential())
                    classifier = getattr(self._model, self._classification_layer[0])
                    for layer in extra_layers:
                        classifier.append(torch.nn.Linear(num_features, layer))
                        classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                    classifier.append(torch.nn.Linear(num_features, num_classes))
                else:
                    classifier = torch.nn.Sequential(torch.nn.Linear(num_features, num_classes))
                    setattr(self._model, self._classification_layer[0], classifier)

            self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

            if ipex_optimize and not self._distributed:
                import intel_extension_for_pytorch as ipex
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)
        self._num_classes = num_classes
        return self._model, self._optimizer

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, extra_layers=None, ipex_optimize=False,
              distributed=False, hostfile=None, nnodes=1, nproc_per_node=1):
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
                    at the end of each epoch.
                early_stopping (bool): Enable early stopping if convergence is reached while training
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                seed (int): Optionally set a seed for reproducibility.
                extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a PyTorch model.
                    The input should be a list of integers representing the number and size of the layers,
                    for example [1024, 512] will insert two dense layers, the first with 1024 neurons and the
                    second with 512 neurons.
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

        self._distributed = distributed

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("The extra_layers parameter must be a list of ints but found a list "
                                        "containing {}".format(type(layer)))

        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._set_seed(seed)

        # IPEX optimization can be suppressed with input ipex_optimize=False or
        # If are loading weights, the state dicts need to be loaded before calling ipex.optimize, so get the model
        # from torchvision, but hold off on the ipex optimize call.
        optimize = ipex_optimize and (False if initial_checkpoints else True)

        self._model, self._optimizer = self._get_hub_model(dataset_num_classes, ipex_optimize=optimize,
                                                           extra_layers=extra_layers)

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Call ipex.optimize now, since we didn't call it from _get_hub_model()
            if ipex_optimize and not distributed:
                import intel_extension_for_pytorch as ipex
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)

        if distributed:
            # TODO: for distributed
            # self.export_for_distributed(TLT_DISTRIBUTED_DIR, dataset)
            batch_size = dataset._preprocessed['batch_size']
            self._fit_distributed(hostfile, nnodes, nproc_per_node, epochs, batch_size, ipex_optimize)
        else:
            self._model.train()
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

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            print("The model has not been trained yet, so evaluation is being done using the original model ",
                  "and its classes")
            model = self._model_downloader(self._model_name)
            optimizer = self._optimizer_class(model.parameters(), lr=self._learning_rate)
            # We shouldn't need ipex.optimize() for evaluation
        else:
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

        if self._model is None:
            print("The model has not been trained yet, so predictions are being done using the original model")
            model = self._model_downloader(self._model_name)
            predictions = model(input_samples)
        else:
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
