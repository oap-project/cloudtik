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
# Apache-2.0
#

import os
import dill
import time

import tensorflow as tf
import tensorflow_text  # noqa: F401
import numpy as np

# import horovod
import horovod.tensorflow.keras as hvd

from pydoc import locate


class TrainArguments:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = dict(kwargs)


class Trainer:
    def __init__(self) -> None:
        pass

    def run(self, training_args: TrainArguments):
        hvd.init()

        model = training_args.model
        optimizer_config = training_args.optimizer.get_config()
        loss = training_args.loss

        legacy_optimizer_class = locate('tensorflow.keras.optimizers.legacy.{}'.format(optimizer_config['name']))
        legacy_optimizer_config = legacy_optimizer_class().get_config()
        legacy_optimizer = legacy_optimizer_class.from_config(
            {k: v for k, v in optimizer_config.items() if k in legacy_optimizer_config})

        optimizer = legacy_optimizer

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        if training_args.scaling.lower() == 'weak':
            multiplier = np.sqrt(training_args.batch_size // training_args.batch_denom)
            optimizer.lr = optimizer.lr * multiplier
            batch_size = training_args.batch_size
        elif training_args.scaling.lower() == 'strong':
            optimizer.lr = optimizer.lr * hvd.size()
            batch_size = training_args.batch_size // hvd.size()

        if training_args.category == 'image_classification':
            hvd_optimizer = hvd.DistributedOptimizer(
                optimizer, backward_passes_per_step=5, average_aggregated_gradients=True)
        elif training_args.category == 'text_classification':
            hvd_optimizer = hvd.DistributedOptimizer(
                optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)

        model.compile(
            loss=loss,
            optimizer=hvd_optimizer,
            metrics=['acc'],
            experimental_run_tf_function=False
        )

        warmup = 3
        if hvd.size() == 1:
            warmup = 1

        callbacks = []
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        # Horovod: average metrics among workers at the end of every epoch.
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final accuracy.
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=optimizer.lr, warmup_epochs=warmup, verbose=1))

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            checkpoint_file = os.path.join(
                training_args.objects_path, 'model_checkpoints')
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_file, save_weights_only=False, monitor='val_acc',
                mode='max', save_best_only=True)
            callbacks.append(model_checkpoint_callback)

        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0

        start = time.time()
        steps_per_epoch_per_worker = len(training_args.train_data) // batch_size
        steps_per_epoch_per_worker = steps_per_epoch_per_worker // hvd.size()
        if hvd.size() > 2:
            steps_per_epoch_per_worker += 1
        if steps_per_epoch_per_worker == 0:
            steps_per_epoch_per_worker = 1
        self.history = model.fit(
            training_args.train_data,
            validation_data=training_args.val_data,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch_per_worker,
            epochs=training_args.epochs,
            initial_epoch=0,
            verbose=verbose
        )
        end = time.time()
        if hvd.rank() == 0:
            print("Total elapsed time in seconds = ", end - start)
            print("Total elapsed time in minutes = ", ((end - start) / 60))
            print("Total epochs = ", len(self.history.history['loss']))
            print("Time per epoch in seconds = ", ((end - start) / len(self.history.history['loss'])))
            print("Maximum validation accuracy = ", np.max(self.history.history['acc']))

    @classmethod
    def load_objects(cls, objects_path):
        # Load the saved_model.pb
        model = tf.keras.models.load_model(filepath=objects_path, compile=False)

        # Load the optimizer and restore its state
        checkpoint = tf.train.Checkpoint(optimizer=tf.optimizers.Adam())
        checkpoint.restore(os.path.join(objects_path, 'saved_optimizer-1'))

        # Load the saved loss class name and instantiate the loss
        with open(os.path.join(objects_path, 'saved_loss'), 'rb') as f:
            loss_class, loss_args = dill.load(f)

        # load the dataset(s)
        train_data = tf.data.Dataset.load(os.path.join(objects_path, 'train_data'))
        val_data_path = os.path.join(objects_path, 'val_data')
        if os.path.exists(val_data_path):
            val_data = tf.data.Dataset.load(val_data_path)
        else:
            val_data = None

        if loss_class is None:
            dataset = train_data.unbatch()
            dataset = list(dataset.as_numpy_iterator())
            labels = list()
            for _, label in dataset:
                labels.append(label)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) if len(set(labels)) == 2 else \
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = loss_class(**loss_args)

        return model, checkpoint.optimizer, loss, train_data, val_data
