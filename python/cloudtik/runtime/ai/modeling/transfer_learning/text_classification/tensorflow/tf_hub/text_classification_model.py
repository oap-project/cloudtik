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
import tensorflow as tf

from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.models \
    import ModelDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils \
    import read_json_file
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.tensorflow.text_classification_model \
    import TensorflowTextClassificationModel
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.text_classification_dataset \
    import TextClassificationDataset

# Note that tensorflow_text isn't used directly but the import is required to register ops used by the
# BERT text preprocessor
import tensorflow_text  # noqa: F401


class TFHubTextClassificationModel(TensorflowTextClassificationModel):
    """
    Class to represent a TF Hub pretrained model that can be used for binary text classification
    fine tuning.
    """

    def __init__(self, model_name: str, **kwargs):
        this_dir = os.path.dirname(__file__)
        model_map = read_json_file(os.path.join(
            this_dir, "text_classification_models.json"))
        if model_name not in model_map.keys():
            raise ValueError("The specified TF Hub text classification model ({}) "
                             "is not supported.".format(model_name))

        self._hub_preprocessor = model_map[model_name]["preprocessor"]
        self._model_url = model_map[model_name]["encoder"]
        self._checkpoint_zip = model_map[model_name]["checkpoint_zip"]

        # extra properties that should become configurable in the future
        self._dropout_layer_rate = 0.1
        self._epsilon = 1e-08
        self._generate_checkpoints = True

        # placeholder for model definition
        self._model = None
        self._num_classes = None

        TensorflowTextClassificationModel.__init__(self, model_name, **kwargs)

    @property
    def model_url(self):
        """
        The public URL used to download the TFHub model
        """
        return self._model_url

    @property
    def preprocessor_url(self):
        return self._hub_preprocessor

    @property
    def num_classes(self):
        return self._num_classes

    def _get_hub_model(self, num_classes, extra_layers=None):
        if not self._model:
            input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input_layer')
            preprocessor = ModelDownloader(
                self._hub_preprocessor, source='tf_hub', model_dir=None, name='preprocessing')
            preprocessing_layer = preprocessor.download()
            encoder_inputs = preprocessing_layer(input_layer)
            encoder = ModelDownloader(
                self._model_url, source='tf_hub', model_dir=None, trainable=True, name='encoder')
            encoder_layer = encoder.download()
            outputs = encoder_layer(encoder_inputs)
            net = outputs['pooled_output']
            self._model = tf.keras.Sequential(tf.keras.Model(input_layer, net))

            if extra_layers:
                for layer_size in extra_layers:
                    self._model.add(tf.keras.layers.Dense(layer_size, "relu"))

            if self._dropout_layer_rate is not None:
                self._model.add(tf.keras.layers.Dropout(self._dropout_layer_rate))

            dense_layer_dims = num_classes

            # For binary classification we only need 1 dimension
            if num_classes == 2:
                dense_layer_dims = 1

            self._model.add(tf.keras.layers.Dense(dense_layer_dims, activation=None, name='classifier'))

            self._model.summary(print_fn=print)

        self._num_classes = num_classes
        return self._model

    def train(self, dataset: TextClassificationDataset, output_dir, *,
              epochs=1, initial_checkpoints=None, do_eval=True,
              early_stopping=False, lr_decay=True, seed=None,
              enable_auto_mixed_precision=None, shuffle_files=True, extra_layers=None,
              distributed=False, nnodes=1, nproc_per_node=1, hosts=None, hostfile=None,
              shared_dir=None, temp_dir=None,
              **kwargs):
        """
           Trains the model using the specified binary text classification dataset. If a path to initial checkpoints is
           provided, those weights are loaded before training.

           Args:
               dataset (TextClassificationDataset): The dataset to use for training. If a train subset has been
                                                    defined, that subset will be used to fit the model. Otherwise, the
                                                    entire non-partitioned dataset will be used.
               output_dir (str): A writeable output directory to write checkpoint files during training
               epochs (int): The number of training epochs [default: 1]
               initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
               do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
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
               extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a TFHub model. The input should be a list of
                    integers representing the number and size of the layers, for example [1024, 512] will insert two
                    dense layers, the first with 1024 neurons and the second with 512 neurons.
               seed (int): Optionally set a seed for reproducibility.
               distributed (bool): Boolean flag to use distributed training. Defaults to False.
               nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
               nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
               to 1.
               hosts (str): hosts list for distributed training. Defaults to None.
               hostfile (str): Name of the hostfile for distributed training. Defaults to None.
               shared_dir (str): The shared data dir for distributed training.
               temp_dir (str): The temp data dir at local.

           Returns:
               History object from the model.fit() call

           Raises:
               FileExistsError if the output directory is a file
               TypeError if the dataset specified is not a TextClassificationDataset
               TypeError if the output_dir parameter is not a string
               TypeError if the epochs parameter is not a integer
               TypeError if the initial_checkpoints parameter is not a string
               TypeError if the extra_layers parameter is not a list of integers
        """
        self._check_train_inputs(
            output_dir, dataset, TextClassificationDataset,
            epochs, initial_checkpoints)

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("extra_layers must be a list of ints but found a list containing {}".format(
                            type(layer)))

        dataset_num_classes = len(dataset.class_names)

        self._set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        self._model = self._get_hub_model(dataset_num_classes, extra_layers)
        print("Num dataset classes: ", dataset_num_classes)

        callbacks, train_data, val_data = self._get_train_callbacks(
            dataset, output_dir, initial_checkpoints, do_eval,
            early_stopping, lr_decay, dataset_num_classes)

        if distributed:
            objects_path = self.save_objects(
                train_data, val_data,
                shared_dir, temp_dir)
            self._fit_distributed(
                epochs, shuffle_files,
                nnodes, nproc_per_node, hosts, hostfile,
                objects_path, kwargs.get('use_horovod'))
            self.cleanup_objects(objects_path)
        else:
            history = self._model.fit(
                train_data, validation_data=val_data, epochs=epochs,
                shuffle=shuffle_files, callbacks=callbacks)

            self._history = history.history

            return self._history

    def evaluate(self, dataset: TextClassificationDataset, use_test_set=False):
        """
           If there is a validation set, evaluation will be done on it (by default) or on the test set (by setting
           use_test_set=True). Otherwise, the entire non-partitioned dataset will be used for evaluation.

           Args:
               dataset (TextClassificationDataset): The dataset to use for evaluation.
               use_test_set (bool): Specify if the test partition of the dataset should be used for evaluation.
                                    [default: False)

           Returns:
               Dictionary with loss and accuracy metrics

           Raises:
               TypeError if the dataset specified is not a TextClassificationDataset
               ValueError if the use_test_set=True and no test subset has been defined in the dataset.
               ValueError if the model has not been trained or loaded yet.
        """
        if not isinstance(dataset, TextClassificationDataset):
            raise TypeError("The dataset must be a TextClassificationDataset but found a {}".format(type(dataset)))

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
            raise ValueError("The model must be trained or loaded before evaluation.")

        return self._model.evaluate(eval_dataset)

    def predict(self, input_samples):
        """
           Generates predictions for the specified input samples.

           Args:
               input_samples (str, list, numpy array, tensor, tf.data dataset or a generator keras.utils.Sequence):
                    Input samples to use to predict. These will be sent to the tf.keras.Model predict() function.

           Returns:
               Numpy array of scores

           Raises:
               ValueError if the model has not been trained or loaded yet.
               ValueError if there is a mismatch between the input_samples and the model's expected input.
        """
        if self._model is None:
            raise ValueError("The model must be trained or loaded before predicting.")

        # If a single string is passed in, make it a list so that it's compatible with the keras model predict
        if isinstance(input_samples, str):
            input_samples = [input_samples]

        return tf.sigmoid(self._model.predict(input_samples)).numpy()
