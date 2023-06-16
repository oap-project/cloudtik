import argparse
import json
import os
from pathlib import Path

import yaml
from transformers import (
    TrainingArguments
)

from train_vision import (
    run as run_train_vision,
    TrainerArguments as VisionTrainerArguments
)
from sentiment_analysis.run import run as run_train_doc
from sentiment_analysis.utils import (
    TrainerArguments as DocTrainerArguments,
    DatasetConfig
)
from data.split_data import get_split_images_output_dir, get_split_doc_output_dir


def _load_config_from_file(config, config_file):
    if config_file.endswith(".json"):
        args_in_file = json.loads(Path(config_file).read_text())
    else:
        # default a yaml file
        with open(config_file, "r") as f:
            args_in_file = yaml.safe_load(f)

    for key in args_in_file:
        setattr(config, key, args_in_file[key])


def run(args):
    this_dir = os.path.dirname(__file__)
    config_dir = os.path.join(
        os.path.dirname(this_dir), "config")
    doc_args = DocTrainerArguments()

    doc_modeling_config_file = os.path.join(config_dir, "doc_modeling_config.yaml")
    _load_config_from_file(doc_args, doc_modeling_config_file)

    doc_args.dataset = "local"
    dataset_config = DatasetConfig()

    # load dataset config from dataset_config file
    doc_dataset_config_file = os.path.join(config_dir, "doc_dataset_config.yaml")
    _load_config_from_file(dataset_config, doc_dataset_config_file)

    split_doc_output_dir = get_split_doc_output_dir(args.processed_data_path)
    dataset_config.train = os.path.join(split_doc_output_dir, "train.csv")
    dataset_config.test = os.path.join(split_doc_output_dir, "test.csv")
    doc_args.dataset_config = dataset_config

    training_args = TrainingArguments()

    # load training arguments or set automatically
    doc_training_arguments_file = os.path.join(config_dir, "doc_training_arguments.yaml")
    _load_config_from_file(training_args, doc_training_arguments_file)

    training_args.output_dir = args.output_dir
    training_args.do_predict = (not args.no_predict)

    doc_args.training_args = training_args

    run_train_doc(doc_args)

    # run vision and doc training
    vision_args = VisionTrainerArguments()

    split_images_path = get_split_images_output_dir(args.processed_data_path)
    vision_args.data_path = split_images_path
    vision_args.output_dir = args.output_dir
    vision_args.no_predict = args.no_predict

    run_train_vision(vision_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disease Prediction Training")

    parser.add_argument(
        "--no-predict", "--no_predict",
        default=False, action="store_true",
        help="whether to predict on test the data.")
    parser.add_argument(
        "--processed-data-path", "--processed_data_path",
        type=str,
        help="Path to the processed data directory",
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--temp-dir", "--temp_dir",
        type=str,
        help="Path to the intermediate directory",
    )

    args = parser.parse_args()
    print(args)

    run(args)
