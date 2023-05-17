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
import numpy as np

import tensorflow as tf

from ...image_classification_dataset import ImageClassificationDataset
from ....common.downloader.models import ModelDownloader
from ....common.utils import read_json_file
from ..image_classification_model import \
    TensorflowImageClassificationModel


class TFHubImageClassificationModel(TensorflowImageClassificationModel):
    """
    Class to represent a TF Hub pretrained model for image classification
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Class constructor
        """
        this_dir = os.path.dirname(__file__)
        model_map = read_json_file(os.path.join(
            this_dir, "image_classification_models.json"))
        if model_name not in model_map.keys():
            raise ValueError("The specified TF Hub image classification model ({}) "
                             "is not supported.".format(model_name))

        self._model_url = model_map[model_name]["imagenet_model"]
        self._feature_vector_url = model_map[model_name]["feature_vector"]

        TensorflowImageClassificationModel.__init__(
            self, model_name=model_name, model=None, **kwargs)

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._image_size = model_map[model_name]["image_size"]

    @property
    def model_url(self):
        """
        The public URL used to download the TFHub model
        """
        return self._model_url

    @property
    def feature_vector_url(self):
        """
        The public URL used to download the headless TFHub model used for transfer learning
        """
        return self._feature_vector_url

    def _model_downloader(self, model_name, include_top=False):
        url = self.feature_vector_url if not include_top else self.model_url
        downloader = ModelDownloader(url, source='tf_hub', model_dir=None,
                                     input_shape=(self.image_size, self.image_size, 3),
                                     trainable=self.do_fine_tuning)
        model = downloader.download()
        return tf.keras.Sequential(model)

    def _get_hub_model(self, num_classes, extra_layers=None):

        if not self._model:
            self._model = self._model_downloader(self._model_name)

            if extra_layers:
                for layer_size in extra_layers:
                    self._model.add(tf.keras.layers.Dense(layer_size, "relu"))

            if self.dropout_layer_rate is not None:
                self._model.add(tf.keras.layers.Dropout(self.dropout_layer_rate))

            self._model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, enable_auto_mixed_precision=None,
              shuffle_files=True, extra_layers=None, distributed=False, hostfile=None,
              nnodes=1, nproc_per_node=1, **kwargs):
        """
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the feature extractor layer from TF Hub and add on a dense layer based on the number of classes
            in the specified dataset. The model is compiled and trained for the specified number of epochs. If a
            path to initial checkpoints is provided, those weights are loaded before training.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for checkpoint files
                epochs (int): Number of epochs to train the model (default: 1)
                initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                early_stopping (bool): Enable early stopping if convergence is reached while training
                    at the end of each epoch.
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms.
                shuffle_files (bool): Boolean specifying whether to shuffle the training data before each epoch.
                seed (int): Optionally set a seed for reproducibility.
                extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a TFHub model. The input should be a list of
                    integers representing the number and size of the layers, for example [1024, 512] will insert two
                    dense layers, the first with 1024 neurons and the second with 512 neurons.

            Returns:
                History object from the model.fit() call

            Raises:
               FileExistsError if the output directory is a file
               TypeError if the dataset specified is not an ImageClassificationDataset
               TypeError if the output_dir parameter is not a string
               TypeError if the epochs parameter is not a integer
               TypeError if the initial_checkpoints parameter is not a string
               TypeError if the extra_layers parameter is not a list of integers
        """

        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints)

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("The extra_layers parameter must be a list of ints",
                                        "but found a list containing {}".format(type(layer)))
        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        self._model = self._get_hub_model(dataset_num_classes, extra_layers)

        callbacks, train_data, val_data = self._get_train_callbacks(dataset, output_dir, initial_checkpoints, do_eval,
                                                                    early_stopping, lr_decay)

        if distributed:
            self.export_for_distributed(train_data, val_data)
            self._fit_distributed(epochs, shuffle_files, hostfile, nnodes, nproc_per_node, kwargs.get('use_horovod'))
            self.cleanup_saved_objects_for_distributed()
        else:
            history = self._model.fit(train_data, epochs=epochs, shuffle=shuffle_files, callbacks=callbacks,
                                      validation_data=val_data)
            self._history = history.history
            return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False):
        """
        Evaluate the accuracy of the model on a dataset.

        If there is a validation subset, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_dataset = dataset.test_subset
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_dataset = dataset.validation_subset
        else:
            eval_dataset = dataset.dataset

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            print("The model has not been trained yet, so evaluation is being done using the original model ",
                  "and its classes")
            original_model = self._model_downloader(self._model_name, include_top=True)
            original_model.compile(
                optimizer=self._optimizer_class(),
                loss=self._loss,
                metrics=['acc'])
            return original_model.evaluate(eval_dataset)
        else:
            return self._model.evaluate(eval_dataset)

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
            original_model = self._model_downloader(self._model_name, include_top=True)
            predictions = original_model.predict(input_samples)
        else:
            predictions = self._model.predict(input_samples)
        if return_type == 'class':
            return np.argmax(predictions, axis=-1)
        elif return_type == 'probabilities':
            return tf.nn.softmax(predictions)
        else:
            return predictions
