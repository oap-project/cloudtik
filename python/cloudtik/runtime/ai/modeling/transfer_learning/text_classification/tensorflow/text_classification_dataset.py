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

from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.text_classification_dataset \
    import TextClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.common.tensorflow.dataset import TensorflowDataset


class TensorflowTextClassificationDataset(TextClassificationDataset, TensorflowDataset):
    """
    A user custom text classification dataset that can be used with TensorFlow models.
    Note that this dataset class expects a .csv file with two columns where the first column is the label and
    the second column is the text/sentence to classify.

    For example, a comma separated value file will look similar to the snippet below:

    .. code-block:: text

        class_a,<text>
        class_b,<text>
        class_a,<text>
        ...

    If the .csv files has more columns, the select_cols or exclude_cols parameters can be used to filter out which
    columns will be parsed.

    Args:
        dataset_dir (str): Directory containing the dataset
        dataset_name (str): Name of the dataset. If no dataset name is given, the dataset_dir folder name
                            will be used as the dataset name.
        csv_file_name (str): Name of the csv file to load from the dataset directory
        class_names (list): List of ordered class names
        label_map_func (function): optional; Maps the label_map_func across the label column of the dataset to apply a
                                   transform to the elements. For example, if the .csv file has string class labels
                                   instead of numerical values, provide a function that maps the string to a numerical
                                   value.
        defaults (list): optional; List of default values for the .csv file fields. Defaults to [tf.string, tf.string]
        delimiter (str): optional; String character that separates the label and text in each row. Defaults to ",".
        header (bool): optional; Boolean indicating whether or not the csv file has a header line that should be
                       skipped. Defaults to False.
        select_cols (list): optional; Specify a list of sorted indices for columns from the dataset file(s) that should
                            be parsed. Defaults to parsing all columns. At most one of select_cols and exclude_cols can
                            be specified.
        exclude_cols (list): optional; Specify a list of sorted indices for columns from the dataset file(s) that should
                             be excluded from parsing. Defaults to parsing all columns. At most one of select_cols and
                             exclude_cols can be specified.
        shuffle_files (bool): optional; Whether to shuffle the data. Defaults to True.
        seed (int): optional; Random seed for shuffling

    Raises:
        FileNotFoundError if the csv file is not found in the dataset directory
        TypeError if the class_names parameter is not a list or the label_map_func is not callable
        ValueError if the class_names list is empty

    """

    def __init__(self, dataset_dir, dataset_name, csv_file_name, class_names, label_map_func=None,
                 defaults=[tf.string, tf.string], delimiter=",", header=False, select_cols=None, exclude_cols=None,
                 shuffle_files=True, seed=None, **kwargs):
        """
        Class constructor
        """
        dataset_file = os.path.join(dataset_dir, csv_file_name)
        if not os.path.exists(dataset_file):
            raise FileNotFoundError("The dataset file ({}) does not exist".format(dataset_file))

        if not isinstance(class_names, list):
            raise TypeError("The class_names is expected to be a list, but found a {}", type(class_names))
        if len(class_names) == 0:
            raise ValueError("The class_names list cannot be empty.")

        if label_map_func and not callable(label_map_func):
            raise TypeError("The label_map_func is expected to be a function, but found a {}", type(label_map_func))

        # The dataset name is only used for informational purposes. Default to use the file name without extension.
        if not dataset_name:
            dataset_name = csv_file_name[:csv_file_name.index('.')] if '.' in csv_file_name else csv_file_name

        TextClassificationDataset.__init__(self, dataset_dir, dataset_name)

        self._dataset = tf.data.experimental.CsvDataset(filenames=dataset_file,
                                                        record_defaults=defaults,
                                                        field_delim=delimiter,
                                                        use_quote_delim=False,
                                                        header=header,
                                                        select_cols=select_cols,
                                                        exclude_cols=exclude_cols)

        if shuffle_files:
            self._dataset = self._dataset.shuffle(1, seed=seed)

        # Count the number of lines in the csv file to get the dataset length
        dataset_len = sum(1 for _ in open(dataset_file))

        if header:
            dataset_len -= 1

        # Set the cardinality so that the dataset length can be used for shuffle splits and progress bars
        self._dataset = self._dataset.apply(tf.data.experimental.assert_cardinality(dataset_len))

        # If a map function has not been defined, we know that we a least need to convert the string from the
        # csv file to a integer for the label field
        if not label_map_func:
            def label_map_func(x):
                return int(x)

        self._dataset = self._dataset.map(lambda x, y: (y, label_map_func(x)))

        self._info = {
            "name": dataset_name,
            "dataset_dir": dataset_dir,
            "file_name": csv_file_name,
            "delimiter": delimiter,
            "defaults": defaults,
            "header": header,
            "select_cols": select_cols,
            "exclude_cols": exclude_cols
        }
        self._preprocessed = None

        self._class_names = class_names
        self._train_pct = 1.0
        self._val_pct = 0
        self._test_pct = 0
        self._validation_type = None
        self._train_subset = None
        self._validation_subset = None
        self._test_subset = None

    @property
    def class_names(self):
        """
        Returns the list of class names
        """
        return self._class_names

    @property
    def info(self):
        """
        Returns a dictionary of information about the dataset
        """
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        """
        Returns the framework dataset object (tf.data.Dataset)
        """
        return self._dataset

    def preprocess(self, batch_size):
        """
            Batch the dataset

            Args:
                batch_size (int): desired batch size

            Raises:
                TypeError if the batch_size is not a positive integer
                ValueError if the dataset is not defined or has already been processed
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be a positive integer")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        # Get the non-None splits
        split_list = ['_dataset', '_train_subset', '_validation_subset', '_test_subset']
        subsets = [s for s in split_list if getattr(self, s, None)]
        for subset in subsets:
            setattr(self, subset, getattr(self, subset).cache())
            setattr(self, subset, getattr(self, subset).batch(batch_size))
            setattr(self, subset, getattr(self, subset).prefetch(tf.data.AUTOTUNE))
        self._preprocessed = {'batch_size': batch_size}
