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
from typing import Optional, Any, Union, Dict, List
import yaml
import numpy as np
from scipy.special import softmax

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
class TrainerArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    smoke_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to execute in sanity check mode."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker testing, truncate the number of testing examples to this "
                    "value if set."
        },
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    multi_instance: bool = field(
        default=False,
        metadata={
            "help": "Whether to use multi-instance mode"
        },
    )
    instance_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "for multi-instance inference, to indicate which instance this is."
        },
    )
    dataset: Optional[str] = field(
        default='imdb',
        metadata={
            "help": "Select dataset ('imdb' / 'sst2' / local). Default is 'imdb'"
        },
    )
    dataset_config: Optional[Union[str, dict]] = field(
        default=None,
        metadata={
            "help": "The dict or file for dataset config (typically for local)"
        },
    )
    training_args: Optional[Union[str, dict]] = field(
        default=None,
        metadata={
            "help": "The dict or file for training arguments"
        },
    )


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


class PredsLabels:
    def __init__(self, preds, labels):
        self.predictions = preds
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
    return "\n\n******** TEST METRICS ********\n" + '\n'.join(f'{k}: {v}' for k, v in metrics.items())


def save_performance_metrics(trainer, data, output_file):
    label_map = {i:v for i,v in enumerate(data.features['label'].names)}
    predictions = trainer.predict(data)

    predictions_report = {}
    predictions_report["label_id"] = [label_map[i] for i in predictions.label_ids.tolist()]
    predictions_report["predictions_label"] = [label_map[i] for i in np.argmax(predictions.predictions, axis=1).tolist() ]
    predictions_report["predictions_probabilities"] = softmax(predictions.predictions, axis=1).tolist() 
    predictions_report["metrics"] = predictions.metrics
    
    with open(output_file, 'w') as file:
        _ = yaml.dump(predictions_report, file) 


def parse_arguments(parser, arguments):
    if isinstance(arguments, str):
        if arguments.endswith(".json"):
            args = parser.parse_json_file(json_file=arguments)
        else:
            # default a yaml file
            with open(arguments, "r") as f:
                args_in_yaml = yaml.safe_load(f)
            args = parser.parse_dict(args=args_in_yaml)
    elif isinstance(arguments, dict):
        # a dict
        args = parser.parse_dict(args=arguments)
    else:
        # it is already an arguments object
        return arguments

    return args
