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

import yaml
from transformers import (
    HfArgumentParser,
    TrainingArguments
)
from transformers import logging as hf_logging

from disease_prediction.dlsa.trainer import Trainer
from disease_prediction.dlsa.predictor import Predictor
from disease_prediction.dlsa.utils import TrainerArguments, parse_arguments, DatasetConfig

from cloudtik.runtime.ai.util.utils import load_config_from

hf_logging.set_verbosity_info()


def _run(args):
    if not args.no_train:
        if not args.model_dir:
            args.model_dir = args.training_args.output_dir

        training_args = copy.deepcopy(args.training_args)
        training_args.do_train = True
        training_args.do_predict = True

        kwargs = {"args": args, "training_args": training_args}
        trainer = Trainer(**kwargs)
        trainer.train()

    if not args.no_predict:
        # for predict set the model name or path to model dir if there is one
        _args = copy.deepcopy(args)
        if args.model_dir:
            _args.model_name_or_path = args.model_dir

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
