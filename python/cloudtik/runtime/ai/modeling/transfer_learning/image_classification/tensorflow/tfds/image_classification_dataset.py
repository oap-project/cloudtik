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

import tensorflow as tf

from cloudtik.runtime.ai.modeling.transfer_learning.image_classification.image_classification_dataset \
    import ImageClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.datasets \
    import DataDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.tensorflow.dataset \
    import TensorflowDataset


class TFDSImageClassificationDataset(ImageClassificationDataset, TensorflowDataset):
    """
    An image classification dataset from the TensorFlow datasets catalog
    """
    def __init__(self, dataset_dir, dataset_name, split=["train"],
                 as_supervised=True, shuffle_files=True, seed=None, **kwargs):
        """
        Class constructor
        """
        if not isinstance(split, list):
            raise ValueError("Value of split argument must be a list.")
        ImageClassificationDataset.__init__(self, dataset_dir, dataset_name)
        self._preprocessed = {}
        self._seed = seed
        tf.get_logger().setLevel('ERROR')

        downloader = DataDownloader(
            dataset_name, dataset_dir=dataset_dir,
            source='tfds', as_supervised=as_supervised,
            shuffle_files=shuffle_files, with_info=True)
        data, self._info = downloader.download(split=split)

        self._dataset = None
        self._train_subset = None
        self._validation_subset = None
        self._test_subset = None

        if len(split) == 1:
            self._validation_type = None  # Train & evaluate on the whole dataset
            self._dataset = data[0]
        else:
            self._validation_type = 'defined_split'  # Defined by user or TFDS
            for i, s in enumerate(split):
                if s == 'train':
                    self._train_subset = data[i]
                elif s == 'validation':
                    self._validation_subset = data[i]
                elif s == 'test':
                    self._test_subset = data[i]
                self._dataset = data[i] if self._dataset is None else self._dataset.concatenate(data[i])

    @property
    def class_names(self):
        """Returns the list of class names"""
        return self._info.features["label"].names

    @property
    def info(self):
        """Returns a dictionary of information about the dataset"""
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        """
        Returns the framework dataset object (tf.data.Dataset)
        """
        return self._dataset

    def preprocess(self, image_size, batch_size, add_aug=None):
        """
        Preprocess the dataset to convert to float32, resize, and batch the images

            Args:
                image_size (int): desired square image size
                batch_size (int): desired batch size
                add_aug (None or list[str]): Choice of augmentations (RandomHorizontalandVerticalFlip,
                                             RandomHorizontalFlip, RandomVerticalFlip, RandomZoom, RandomRotation) to
                                             be applied during training

            Raises:
                ValueError if the dataset is not defined or has already been processed
        """
        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be a positive integer")
        if not isinstance(image_size, int) or image_size < 1:
            raise ValueError("image_size should be a positive integer")
        if not (self._dataset or self._train_subset or self._validation_subset or self._test_subset):
            raise ValueError("Unable to preprocess, because the dataset hasn't been defined.")

        if add_aug is not None:
            aug_dict = {
                'hvflip': tf.keras.layers.RandomFlip("horizontal_and_vertical",
                                                     input_shape=(image_size, image_size, 3), seed=self._seed),
                'hflip': tf.keras.layers.RandomFlip("horizontal",
                                                    input_shape=(image_size, image_size, 3), seed=self._seed),
                'vflip': tf.keras.layers.RandomFlip("vertical",
                                                    input_shape=(image_size, image_size, 3), seed=self._seed),
                'rotate': tf.keras.layers.RandomRotation(0.5, seed=self._seed),
                'zoom': tf.keras.layers.RandomZoom(0.3, seed=self._seed)}
            aug_list = ['hvflip', 'hflip', 'vflip', 'rotate', 'zoom']

            data_augmentation = tf.keras.Sequential()

            for option in add_aug:
                if option not in aug_list:
                    raise ValueError("Unsupported augmentation for TensorFlow:{}. \
                    Supported augmentations are {}".format(option, aug_list))
                data_augmentation.add(aug_dict[option])

        def preprocess_image(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, image_size, image_size)
            return (image, label)

        # Get the non-None splits
        split_list = ['_dataset', '_train_subset', '_validation_subset', '_test_subset']
        subsets = [s for s in split_list if getattr(self, s, None)]
        for subset in subsets:
            setattr(self, subset, getattr(self, subset).map(preprocess_image))
            setattr(self, subset, getattr(self, subset).cache())
            setattr(self, subset, getattr(self, subset).batch(batch_size))
            setattr(self, subset, getattr(self, subset).prefetch(tf.data.AUTOTUNE))
            if add_aug is not None and subset in ['_dataset', '_train_subset']:
                setattr(self, subset, getattr(self, subset).map(lambda x, y: (data_augmentation(x, training=True), y),
                                                                num_parallel_calls=tf.data.AUTOTUNE))
        self._preprocessed = {'image_size': image_size, 'batch_size': batch_size}
