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

from pydoc import locate

from cloudtik.runtime.ai.modeling.transfer_learning.common.utils import get_framework_name, get_category_name

dataset_map = {
    "image_classification": {
        "pytorch": {
            "torchvision": "TorchvisionImageClassificationDataset",
            "user": "PyTorchImageClassificationDataset",
        },
        "tensorflow": {
            "tfds": "TFDSImageClassificationDataset",
            "user": "TensorflowImageClassificationDataset"
        }
    },
    "text_classification": {
        "pytorch": {
            "hugging_face": "HuggingFaceTextClassificationDataset",
            "user": "PyTorchTextClassificationDataset"
        },
        "tensorflow": {
            "tfds": "TFDSTextClassificationDataset",
            "user": "TensorflowTextClassificationDataset",
        },
    },
    "image_anomaly_detection": {
        "pytorch": {
            "user": "PyTorchImageAnomalyDetectionDataset",
        },
    }
}


def _get_default_dataset_class_name(category, framework):
    framework_name = get_framework_name(framework)
    category_name = get_category_name(category)
    return "{}{}Dataset".format(framework_name, category_name)


def _get_dataset_module_class(category: str, framework: str, source: str = None):
    category = category.lower()
    framework = framework.lower()
    if source is None:
        source = "user"
    if source != "user":
        module = ("cloudtik.runtime.ai.modeling.transfer_learning."
                  "{category}.{framework}.{source}.{category}_dataset").format(
            category=category, framework=framework, source=source)
    else:
        module = ("cloudtik.runtime.ai.modeling.transfer_learning."
                  "{category}.{framework}.{category}_dataset").format(
            category=category, framework=framework)
    if category in dataset_map and (
            framework in dataset_map[category]) and (
            source in dataset_map[category][framework]):
        dataset_class_name = dataset_map[category][framework][source]
    else:
        dataset_class_name = _get_default_dataset_class_name(category, framework)

    return '{}.{}'.format(module, dataset_class_name)


def load_dataset(
        dataset_dir: str, category: str, framework: str, dataset_name=None, **kwargs):
    """A factory method for loading a custom dataset.

    Image classification datasets expect a directory of images organized with subfolders for each image class, which
    can themselves be in split directories named 'train', 'validation', and/or 'test'. Each class subfolder should
    contain .jpg images for the class. The name of the subfolder will be used as the class label.

    .. code-block:: text

        dataset_dir
          ├── class_a
          ├── class_b
          └── class_c

    Or:

    .. code-block:: text

        dataset_dir
          ├── train
          |   ├── class_a
          |   ├── class_b
          |   └── class_c
          ├── validation
          |   ├── class_a
          |   ├── class_b
          |   └── class_c
          └── test
              ├── class_a
              ├── class_b
              └── class_c

    Text classification datasets are expected to be a directory with text/csv file with two columns: the label and the
    text/sentence to classify. See the TFCustomTextClassificationDataset documentation for a list of the additional
    kwargs that are used for loading the a text classification dataset file.

    .. code-block:: text

        class_a,<text>
        class_b,<text>
        class_a,<text>
        ...

    Args:
        dataset_dir (str): directory containing the dataset
        category (str): use case or task the dataset will be used to model
        framework (str): framework
        dataset_name (str): optional; name of the dataset used for informational purposes
        kwargs: optional; additional keyword arguments depending on the type of dataset being loaded

    Returns:
        (dataset)

    Raises:
        NotImplementedError if the type of dataset being loaded is not supported

    Example:
        >>> from cloudtik.runtime.ai.modeling.dataset_factory import load_dataset
        >>> data = load_dataset('/tmp/data/flower_photos', 'image_classification', 'tensorflow')
        Found 3670 files belonging to 5 classes.
        >>> data.class_names
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    """
    if not category:
        raise ValueError("Category parameter must be specified.")
    if not framework:
        raise ValueError("Framework parameter must be specified.")

    dataset_module_class = _get_dataset_module_class(category, framework)
    dataset_class = locate(dataset_module_class)
    return dataset_class(dataset_dir, dataset_name, **kwargs)


def get_dataset(dataset_dir: str, category: str, framework: str,
                dataset_name: str = None, source: str = None, **kwargs):
    """
    A factory method for using a dataset from a catalog.

    Args:
        dataset_dir (str): directory containing the dataset or to which the dataset should be downloaded
        category (str): use case or task the dataset will be used to model
        framework (str): framework
        dataset_name (str): optional; name of the dataset
        source (str): optional; source from which to download the dataset. If a dataset name is
                               provided and no dataset source is given, it will default to use tf_datasets
                               for a TensorFlow model, torchvision for PyTorch CV models, and huggingface
                               datasets for PyTorch NLP models or Hugging Face models.
        **kwargs: optional; additional keyword arguments for the framework or dataset_catalog

    Returns:
        (dataset)

    Raises:
        NotImplementedError if the dataset requested is not supported yet

    Example:
        >>> from cloudtik.runtime.ai.modeling.dataset_factory import get_dataset
        >>> data = get_dataset('/tmp/data/', 'image_classification', 'tensorflow', 'tf_flowers', 'tfds')
        >>> sorted(data.class_names)
        ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    """
    if not category:
        raise ValueError("Category parameter must be specified.")
    if not framework:
        raise ValueError("Framework parameter must be specified.")

    if dataset_name and not source:
        # Try to assume a dataset source based on the other information that we have
        if framework == "tensorflow":
            source = "tfds"
        elif framework == "pytorch":
            if category == "image_classification":
                source = "torchvision"
            elif category == "text_classification":
                source = "hugging_face"

    dataset_module_class = _get_dataset_module_class(category, framework, source)
    dataset_class = locate(dataset_module_class)
    return dataset_class(dataset_dir, dataset_name, **kwargs)
