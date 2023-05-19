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
from pydoc import locate

from cloudtik.runtime.ai.modeling.transfer_learning.common.utils \
    import get_framework_name, get_category_name, list_matched_dirs, list_matched_files, read_json_file

"""
    A registry entry can be in the form of:
    {
        "module": "cloudtik.runtime.ai.modeling.transfer_learning.xxx",
        "class": "XXXModel"
    }
    if module is not specified, it will be formatted in the form of:
    cloudtik.runtime.ai.modeling.transfer_learning.{category}.{framework}.{source}.{category}_model
"""
model_map = {
    "image_classification": {
        "pytorch": {
            "torchvision": {"class": "TorchvisionImageClassificationModel"},
            "pytorch_hub": {"class": "PyTorchHubImageClassificationModel"},
            "user": {"class": "PyTorchImageClassificationModel"},
        },
        "tensorflow": {
            "tf_hub": {"class": "TFHubImageClassificationModel"},
            "keras": {"class": "KerasImageClassificationModel"},
            "user": {"class": "TensorflowImageClassificationModel"},
        }
    },
    "text_classification": {
        "pytorch": {
            "hugging_face": {"class": "HuggingFaceTextClassificationModel"},
        },
        "tensorflow": {
            "tf_hub": {"class": "TFHubTextClassificationModel"},
            "user": {"class": "TensorflowTextClassificationModel"},
        },
    },
    "image_anomaly_detection": {
        "pytorch": {
            "torchvision": {"class": "TorchvisionImageAnomalyDetectionModel"},
            "user": {"class": "PyTorchImageAnomalyDetectionModel"},
        },
    }
}


def _get_default_model_class_name(category, framework):
    framework_name = get_framework_name(framework)
    category_name = get_category_name(category)
    return "{}{}Model".format(framework_name, category_name)


def _get_default_model_module_name(category, framework, source):
    if source is None or source != "user":
        module_name = "{category}.{framework}.{source}.{category}_model".format(
            category=category, framework=framework, source=source)
    else:
        module_name = "{category}.{framework}.{category}_model".format(
            category=category, framework=framework)
    return "cloudtik.runtime.ai.modeling.transfer_learning." + module_name


def _get_model_module_class(category: str, framework: str, source: str = None):
    if not category:
        raise ValueError("Category parameter must be specified.")
    if not framework:
        raise ValueError("Framework parameter must be specified.")

    category = category.lower()
    framework = framework.lower()
    source = "user" if source is None else source.lower()
    if category in model_map and (
            framework in model_map[category]) and (
            source in model_map[category][framework]):
        model_registry = model_map[category][framework][source]
        if 'module' in model_registry:
            module = model_registry['module']
        else:
            module = _get_default_model_module_name(category, framework, source)
        model_class_name = model_registry['class']
    else:
        module = _get_default_model_module_name(category, framework, source)
        model_class_name = _get_default_model_class_name(category, framework)

    return '{}.{}'.format(module, model_class_name)


def load_model(model_name: str, model, category: str, framework: str, **kwargs):
    """A factory method for loading an existing model.
        Args:
            model_name (str): name of model
            model (model or str): model object or directory with a saved_model.pb or model.pt file to load
            category (str): the category of the model
            framework (str): framework: pytorch or tensorflow
            kwargs: optional; additional keyword arguments for optimizer and loss function configuration.
                The `optimizer` and `loss` arguments can be set to Optimizer and Loss classes, depending on the model's
                framework (examples: `optimizer=tf.keras.optimizers.Adam` for TensorFlow,
                `loss=torch.nn.CrossEntropyLoss` for PyTorch). Additional keywords for those classes' initialization
                can then be provided to further configure the objects when they are created (example: `amsgrad=True`
                for the PyTorch Adam optimizer). Refer to the framework documentation for the function you want to use.

        Returns:
            model object

        Examples:
            >>> from tensorflow.keras import Sequential, Input
            >>> from tensorflow.keras.layers import Dense
            >>> from cloudtik.runtime.ai.modeling.transfer_learning.model_factory import load_model
            >>> my_model = Sequential([Input(shape=(3,)), Dense(4, activation='relu'), Dense(5, activation='softmax')])
            >>> model = load_model('my_model', my_model, 'image_classification', 'tensorflow')

    """
    model_module_class = _get_model_module_class(category, framework)
    model_class = locate(model_module_class)
    return model_class(model_name, model, **kwargs)


def get_model(model_name: str, category: str, framework: str, source: str = None, **kwargs):
    """A factory method for creating models.

        Args:
            model_name (str): name of model
            category (str): the category of the model
            framework (str): framework: pytorch or tensorflow
            source (str): source of the model
            kwargs: optional; additional keyword arguments for optimizer and loss function configuration.
                The `optimizer` and `loss` arguments can be set to Optimizer and Loss classes, depending on the model's
                framework (examples: `optimizer=tf.keras.optimizers.Adam` for TensorFlow,
                `loss=torch.nn.CrossEntropyLoss` for PyTorch). Additional keywords for those classes' initialization
                can then be provided to further configure the objects when they are created (example: `amsgrad=True`
                for the PyTorch Adam optimizer). Refer to the framework documentation for the function you want to use.

        Returns:
            model object

        Raises:
            NotImplementedError if the model requested is not supported yet

        Example:
            >>> from cloudtik.runtime.ai.modeling.transfer_learning.model_factory import get_model
            >>> model = get_model('efficientnet_b0', 'tensorflow')
            >>> model.image_size
            224
    """
    if not model_name:
        raise ValueError("Model name parameter must be specified.")

    if source is None or category is None or framework is None:
        # if no source specified, find a source with the model name
        # based on the framework and category
        category, framework, source = select_any(
            model_name, category, framework, source)

    model_module_class = _get_model_module_class(category, framework, source)
    model_class = locate(model_module_class)
    return model_class(model_name, **kwargs)


def select_any(model_name, category: str = None, framework: str = None, source: str = None):
    candidate_models = search_model(
        model_name, category, framework, source)
    if not candidate_models:
        raise ValueError("No models found with the specified parameters")
    if category is None:
        category = list(candidate_models.keys())[0]

    framework_models = candidate_models[category]
    if not framework_models:
        raise ValueError("No models found with the specified parameters")
    if framework is None:
        if len(framework_models) > 1:
            raise ValueError("Multiple frameworks found for the model name specified")
        framework = list(framework_models.keys())[0]

    source_models = framework_models[framework]
    if not source_models:
        raise ValueError("No models found with the specified parameters")
    if source is None:
        source = list(source_models.keys())[0]

    return category, framework, source


def search_model(model_name, category: str = None, framework: str = None, source: str = None):
    models = search_models(category, framework, source)
    candidate_models = {}

    for model_category in models.keys():
        if model_name in models[model_category]:
            # Found a matching model
            candidate_models[category] = models[model_category][model_name]

    return candidate_models


def search_models(category: str = None, framework: str = None, source: str = None):
    """
    Search 3 levels of folder: category, framework and source for models json file.
    Returns a dictionary of supported models organized by category, model name, and framework.
    The leaf items in the dictionary are attributes about the pretrained model.
    """
    # Models dictionary with keys for category / model name / framework / source / model info
    models = {}

    # search models with the information
    # list the categories from this dir
    this_dir = os.path.dirname(__file__)
    for category_candidate in list_matched_dirs(this_dir, category):
        category_dir = os.path.join(this_dir, category_candidate)
        for framework_candidate in list_matched_dirs(category_dir, framework):
            framework_dir = os.path.join(category_dir, framework_candidate)
            for source_candidate in list_matched_dirs(framework_dir, source):
                source_dir = os.path.join(framework_dir, source_candidate)
                for models_file in list_matched_files(source_dir, "models.json"):
                    _add_models(
                        models, category_candidate, framework_candidate, source_candidate,
                        os.path.join(source_dir, models_file)
                    )
    return models


def _add_models(
        models, category: str, framework: str, source: str, models_config_file):
    models_map = read_json_file(models_config_file)
    if category not in models:
        models[category] = {}
    models_category = models[category]
    for model_name in models_map.keys():
        if model_name not in models_category:
            models_category[model_name] = {}
        models_model = models_category[model_name]
        if framework not in models_model:
            models_model[framework] = {}
        models_framework = models_model[framework]
        models_framework[source] = models_map[model_name]
