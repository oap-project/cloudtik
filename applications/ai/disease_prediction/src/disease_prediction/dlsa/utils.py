# Copyright (C) 2022 Intel Corporation                                                                                              
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Optional, Dict
import yaml
import numpy as np
from scipy.special import softmax

from datasets import ClassLabel

SEC_TO_NS_SCALE = 1000000000


@dataclass
class Benchmark:
    summary_msg: str = field(default_factory=str)

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self, step):
        start = perf_counter_ns()
        yield
        ns = perf_counter_ns() - start
        msg = f"\n{'*' * 70}\n'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        # print(msg)
        self.summary_msg += msg + '\n'

    def summary(self):
        print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


@dataclass
class DatasetConfig:
    """
    Configuration for dataset
    """
    train: str = field(
        default=None,
        metadata={"help": "Path to the train data"}
    )
    test: str = field(
        default=None,
        metadata={"help": "Path to the test data"}
    )
    delimiter: Optional[str] = field(
        default=',',
        metadata={
            "help": "delimiter"
        },
    )
    features: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "The features"
        },
    )
    label_list: Optional[list] = field(
        default=None,
        metadata={
            "help": "the list of labels"
        },
    )


class PredictionsLabels:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.label_ids = labels


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def save_train_metrics(train_result, trainer, max_train):
    # pytorch only
    if train_result:
        metrics = train_result.metrics
        metrics["train_samples"] = max_train
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def save_test_metrics(metrics, max_test, output_dir):
    metrics['test_samples'] = max_test
    with open(Path(output_dir) / 'test_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return "\n\n******** TEST METRICS ********\n" + '\n'.join(
        f'{k}: {v}' for k, v in metrics.items())


def label_to_id_mapping(class_labels):
    return {class_labels[i]: i for i in range(len(class_labels))}


def id_to_label_mapping(class_labels):
    if class_labels is None:
        raise ValueError("No class labels information specified.")
    return {i: v for i, v in enumerate(class_labels)}


def save_predictions(
        predictions, data, output_file,
        class_labels=None):
    if class_labels is None:
        # try get from the data feature definition (for train data, it has the label)
        # a better way is to find a ClassLabel feature
        for name, feature in data.features.items():
            if isinstance(feature, ClassLabel):
                class_labels = feature.names
    label_map = id_to_label_mapping(class_labels)

    predictions_report = {}

    if predictions.label_ids is not None:
        predictions_report["label_id"] = [label_map[i] for i in predictions.label_ids.tolist()]
    predictions_report["predictions_label"] = [label_map[i] for i in np.argmax(
        predictions.predictions, axis=1).tolist()]
    predictions_report["predictions_probabilities"] = softmax(predictions.predictions, axis=1).tolist() 
    predictions_report["metrics"] = predictions.metrics
    
    with open(output_file, 'w') as file:
        _ = yaml.dump(predictions_report, file) 
