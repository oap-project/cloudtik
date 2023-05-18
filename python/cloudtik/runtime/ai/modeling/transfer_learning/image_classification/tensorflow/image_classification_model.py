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
import numpy as np
import yaml

import tensorflow as tf

from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.tensorflow.user_image_classification_dataset \
    import UserImageClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_dataset \
    import ImageClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_model \
    import ImageClassificationModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.tensorflow.model \
    import TensorflowModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils \
    import validate_model_name, verify_directory


class TensorflowImageClassificationModel(ImageClassificationModel, TensorflowModel):
    """
    Class to represent a TF custom pretrained model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        self._image_size = None

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._generate_checkpoints = True

        # placeholder for model definition
        self._num_classes = None

        TensorflowModel.__init__(self, model_name)
        ImageClassificationModel.__init__(
            self, self._image_size, self._do_fine_tuning,
            self._dropout_layer_rate, self._model_name)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        config = {'from_logits': True}
        config.update(kwargs)
        self._optimizer_class = optimizer if optimizer else tf.keras.optimizers.Adam
        self._opt_args = {k: v for k, v in config.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else tf.keras.losses.SparseCategoricalCrossentropy
        self._loss_args = {k: v for k, v in config.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        if model is None:
            self._model = None
        elif isinstance(model, str):
            self.load_from_directory(model)
            self._num_classes = self._model.output.shape[-1]
            self._image_size = self._model.input.shape[1]
        elif isinstance(model, tf.keras.Model):
            self._model = model
            self._num_classes = self._model.output.shape[-1]
            self._image_size = self._model.input.shape[1]
        else:
            raise TypeError("The model input must be a keras Model, string, or None but found a {}".format(type(model)))

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _get_train_callbacks(self, dataset, output_dir, initial_checkpoints, do_eval, early_stopping,
                             lr_decay):
        self._optimizer = self._optimizer_class(learning_rate=self._learning_rate, **self._opt_args)
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=['acc'])

        if initial_checkpoints:
            if os.path.isdir(initial_checkpoints):
                initial_checkpoints = tf.train.latest_checkpoint(initial_checkpoints)

            self._model.load_weights(initial_checkpoints)

        class CollectBatchStats(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_losses = []
                self.batch_acc = []

            def on_train_batch_begin(self, batch, logs=None):
                self.model.reset_metrics()

            def on_train_batch_end(self, batch, logs=None):
                self.batch_losses.append(logs['loss'])
                self.batch_acc.append(logs['acc'])

        batch_stats_callback = CollectBatchStats()

        callbacks = [batch_stats_callback]

        if early_stopping:
            stop_early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            callbacks.append(stop_early_callback)

        # Create a callback for generating checkpoints
        if self._generate_checkpoints:
            valid_model_name = validate_model_name(self.model_name)
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
            verify_directory(checkpoint_dir)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, valid_model_name), save_weights_only=True)
            print("Checkpoint directory:", checkpoint_dir)
            callbacks.append(checkpoint_callback)

        # Create a callback for learning rate decay
        if do_eval and lr_decay:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                verbose=2,
                mode='auto',
                cooldown=1,
                min_lr=0.0000000001))

        if dataset._validation_type == 'shuffle_split':
            train_dataset = dataset.train_subset
        else:
            train_dataset = dataset.dataset

        validation_data = dataset.validation_subset if do_eval else None

        return callbacks, train_dataset, validation_data

    def _fit_distributed(self, epochs, shuffle, hostfile, nnodes, nproc_per_node, use_horovod):
        import subprocess
        # TODO: handle distributed training
        # distributed_vision_script = os.path.join(TLT_DISTRIBUTED_DIR, 'tensorflow', 'run_train_tf.py')
        distributed_vision_script = os.path.join('tensorflow', 'run_train_tf.py')

        if use_horovod:
            run_cmd = 'horovodrun'
        else:
            run_cmd = 'mpirun'

            # mpirun requires these flags to be set
            run_cmd += ' --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0'  # noqa: E501

        hostfile_cmd = ''
        np_cmd = ''
        if os.path.isfile(hostfile):

            hostfile_info = self._parse_hostfile(hostfile)
            node_count = 0
            if sum(hostfile_info['slots']) == 0:
                for ip_addr in hostfile_info['ip_addresses']:
                    hostfile_cmd += '{}:{},'.format(ip_addr, nproc_per_node)
                    node_count += 1
                    if node_count == nnodes:
                        break

            elif sum(hostfile_info['slots']) == nnodes * nproc_per_node:
                for ip_addr, slots in zip(hostfile_info['ip_addresses'], hostfile_info['slots']):
                    hostfile_cmd += '{}:{},'.format(ip_addr, slots)
            else:
                print("WARNING: nproc_per_node and slots in hostfile do not add up. Making equal slots for all nodes")
                for ip_addr in hostfile_info['ip_addresses']:
                    hostfile_cmd += '{}:{},'.format(ip_addr, nproc_per_node)

            hostfile_cmd = hostfile_cmd[:-1]  # Remove trailing comma

            # Final check to correct the `-np` flag's value
            nprocs = nnodes * nproc_per_node
            np_cmd = str(nprocs)
        else:
            raise ValueError("Error: Invalid file \'{}\'".format(hostfile))
        script_cmd = 'python ' + distributed_vision_script
        script_cmd += ' --use_case {}'.format('image_classification')
        script_cmd += ' --epochs {}'.format(epochs)
        if shuffle:
            script_cmd += ' --shuffle'

        bash_command = run_cmd.split(' ') + ['-np', np_cmd, '-H', hostfile_cmd] + script_cmd.split(' ')
        print(' '.join(str(e) for e in bash_command))
        subprocess.run(bash_command)

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, enable_auto_mixed_precision=None,
              shuffle_files=True, distributed=False, hostfile=None, nnodes=1, nproc_per_node=1,
              **kwargs):
        """
        Trains the model using the specified image classification dataset. The model is compiled and trained for
        the specified number of epochs. If a path to initial checkpoints is provided, those weights are loaded before
        training.

        Args:
            dataset (ImageClassificationDataset): Dataset to use when training the model
            output_dir (str): Path to a writeable directory for checkpoint files
            epochs (int): Number of epochs to train the model (default: 1)
            initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                latest checkpoint will be used.
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
            early_stopping (bool): Enable early stopping if convergence is reached while training
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

        Returns:
            History object from the model.fit() call

        Raises:
           FileExistsError if the output directory is a file
           TypeError if the dataset specified is not an ImageClassificationDataset
           TypeError if the output_dir parameter is not a string
           TypeError if the epochs parameter is not a integer
           TypeError if the initial_checkpoints parameter is not a string
           RuntimeError if the number of model classes is different from the number of dataset classes
        """

        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints)

        dataset_num_classes = len(dataset.class_names)

        # Check that the number of classes matches the model outputs
        if dataset_num_classes != self.num_classes:
            raise RuntimeError("The number of model outputs ({}) differs from the number of dataset classes ({})".
                               format(self.num_classes, dataset_num_classes))

        self._set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        callbacks, train_data, val_data = self._get_train_callbacks(dataset, output_dir, initial_checkpoints, do_eval,
                                                                    early_stopping, lr_decay)

        if distributed:
            try:
                self.export_for_distributed(train_data, val_data)
                self._fit_distributed(epochs, shuffle_files, hostfile, nnodes, nproc_per_node,
                                      kwargs.get('use_horovod'))
            except Exception as err:
                print("Error: \'{}\' occured while distributed training".format(err))
            finally:
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

        predictions = self._model.predict(input_samples)
        if return_type == 'class':
            return np.argmax(predictions, axis=-1)
        elif return_type == 'probabilities':
            return tf.nn.softmax(predictions)
        else:
            return predictions

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False,
                              resize_interpolation='bicubic', accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                              exit_policy_max_trials=50, tuning_random_seed=9527,
                              tuning_workspace=''):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters. This is currently only supported for TF custom image classification datasets.

        Args:
            config_file_path (str): Destination path on where to write the .yaml config file.
            dataset (Dataset): A dataset object
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

        # We can setup the a custom dataset to use the ImageFolder dataset option in INC. They don't have a TFDS option,
        # so for now, we only support custom datasets for quantization
        if dataset is not UserImageClassificationDataset and type(dataset) != UserImageClassificationDataset:
            raise NotImplementedError('Quantization has only been implemented for TF image classification models '
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

        if "quantization" in config_template.keys() and "calibration" in config_template["quantization"].keys() and \
           "dataloader" in config_template["quantization"]["calibration"].keys():
            dataloader_configs.append(config_template["quantization"]["calibration"]["dataloader"])

        if "evaluation" in config_template.keys():
            if "accuracy" in config_template["evaluation"].keys() and \
               "dataloader" in config_template["evaluation"]["accuracy"].keys():
                dataloader_configs.append(config_template["evaluation"]["accuracy"]["dataloader"])
            if "performance" in config_template["evaluation"].keys() and \
               "dataloader" in config_template["evaluation"]["performance"].keys():
                dataloader_configs.append(config_template["evaluation"]["performance"]["dataloader"])

        transform_config = {
            "PaddedCenterCrop": {
                "size": self.image_size,
                "crop_padding": 32
            },
            "Resize": {
                "size": self.image_size,
                "interpolation": resize_interpolation
            },
            "Rescale": {}
        }

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
            yaml.dump(config_template, config_file)

    def optimize_graph(self, saved_model_dir, output_dir):
        """
        Performs FP32 graph optimization using the Intel Neural Compressor on the model in the saved_model_dir
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
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir
            FileExistsError if the output_dir already has a saved_model.pb file
        """
        # The saved model directory should exist and contain a saved_model.pb file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                saved_model_dir))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "saved_model.pb")):
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "saved_model.pb"))

        from neural_compressor.experimental import Graph_Optimization

        graph_optimizer = Graph_Optimization()
        graph_optimizer.model = saved_model_dir
        optimized_graph = graph_optimizer()

        # If optimization was successful, save the model
        if optimized_graph:
            optimized_graph.save(output_dir)

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        """
        Performs post training quantization using the Intel Neural Compressor on the model from the saved_model_dir
        using the specified config file. The quantized model is written to the output directory

        Args:
            saved_model_dir (str): Source directory for the model to quantize.
            output_dir (str): Writable output directory to save the quantized model
            inc_config_path (str): Path to an INC config file (.yaml)

        Returns:
            None

        Raises:
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            FileExistsError if the output_dir already has a saved_model.pb file
        """
        # The saved model directory should exist and contain a saved_model.pb file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                saved_model_dir))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "saved_model.pb")):
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "saved_model.pb"))

        from neural_compressor.experimental import Quantization

        quantizer = Quantization(inc_config_path)
        quantizer.model = saved_model_dir
        quantized_model = quantizer.fit()

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)

    def benchmark(self, saved_model_dir, inc_config_path, mode='performance'):
        """
        Use INC to benchmark the specified model for performance or accuracy.

        Args:
            saved_model_dir (str): Path to the directory where the saved model is located
            inc_config_path (str): Path to an INC config file (.yaml)
            mode (str): performance or accuracy (defaults to performance)

        Returns:
            None

        Raises:
            NotADirectoryError if the saved_model_dir is not a directory
            FileNotFoundError if a saved_model.pb is not found in the saved_model_dir or if the inc_config_path file
            is not found.
            ValueError if an unexpected mode is provided
        """
        # The saved model directory should exist and contain a saved_model.pb file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "saved_model.pb")):
            raise FileNotFoundError("The saved model directory ({}) should have a saved_model.pb file".format(
                saved_model_dir))

        # Validate mode
        if mode not in ['performance', 'accuracy']:
            raise ValueError("Invalid mode: {}. Expected mode to be 'performance' or 'accuracy'.".format(mode))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        from neural_compressor.experimental import Benchmark

        evaluator = Benchmark(inc_config_path)
        evaluator.model = saved_model_dir
        return evaluator(mode)
