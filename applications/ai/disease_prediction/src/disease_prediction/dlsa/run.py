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
import copy
import os
import sys
from typing import Optional, Union
import yaml
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    TrainingArguments
)
from transformers import logging as hf_logging

from disease_prediction.utils import DEFAULT_TRAIN_OUTPUT, DEFAULT_PREDICT_OUTPUT

from disease_prediction.dlsa.trainer import Trainer
from disease_prediction.dlsa.predictor import Predictor
from disease_prediction.dlsa.utils import DatasetConfig

from cloudtik.runtime.ai.util.utils import load_config_from
from disease_prediction.utils import parse_arguments

hf_logging.set_verbosity_info()


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

    model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the model dir"
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the output"
        },
    )
    temp_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the intermediate output"
        },
    )
    train_output: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the train output"
        },
    )
    predict_output: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the predict output"
        },
    )

    no_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train."}
    )
    no_predict: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to predict."}
    )


def _run(args):
    if not args.no_train:
        _args = copy.deepcopy(args)
        if not _args.model_dir:
            _args.model_dir = _args.training_args.output_dir

        if not _args.train_output:
            _args.train_output = os.path.join(
                _args.output_dir, DEFAULT_TRAIN_OUTPUT)
        _args.training_args.do_train = True
        _args.training_args.do_predict = True

        kwargs = {"args": _args, "training_args": _args.training_args}
        trainer = Trainer(**kwargs)
        trainer.train()

    if not args.no_predict:
        # for predict set the model name or path to model dir if there is one
        _args = copy.deepcopy(args)
        if args.model_dir:
            _args.model_name_or_path = args.model_dir

        if not _args.predict_output:
            _args.predict_output = os.path.join(
                _args.output_dir, DEFAULT_PREDICT_OUTPUT)
        _args.training_args.do_train = False
        _args.training_args.do_predict = True

        kwargs = {"args": _args, "training_args": _args.training_args}
        predictor = Predictor(**kwargs)
        predictor.predict()


def run(args:TrainerArguments):
    if args.training_args is None:
        args.training_args = TrainingArguments(output_dir=args.output_dir)
    else:
        # load the training arguments
        _parser = HfArgumentParser(TrainingArguments)
        args.training_args = parse_arguments(_parser, args.training_args)

    if args.dataset_config is not None:
        # load the dataset config
        dataset_config = DatasetConfig()
        args.dataset_config = load_config_from(
            args.dataset_config, dataset_config)

    if args.dataset == "local" and args.dataset_config is None:
        raise ValueError("Dataset config is missing for local database.")

    _run(args)


if __name__ == "__main__":
    parser = HfArgumentParser(TrainerArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file = os.path.abspath(sys.argv[1])
        args = parser.parse_json_file(json_file=json_file)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        yaml_file = os.path.abspath(sys.argv[1])
        with open(yaml_file, "r") as f:
            args_in_yaml = yaml.safe_load(f)
        args = parser.parse_dict(args=args_in_yaml)
    else:
        args = parser.parse_args_into_dataclasses()

    print(args)

    run(args)
