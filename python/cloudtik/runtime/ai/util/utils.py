import os
import shutil
from pathlib import Path

import json
import yaml


def load_config_from(source, config=None):
    if isinstance(source, str):
        if source.endswith(".json"):
            args_in_source = json.loads(Path(source).read_text())
        else:
            # default a yaml file
            with open(source, "r") as f:
                args_in_source = yaml.safe_load(f)
    elif isinstance(source, dict):
        # a dict
        args_in_source = source
    else:
        # already some config object
        return source
    if config is not None:
        for key in args_in_source:
            setattr(config, key, args_in_source[key])
        return config
    else:
        return args_in_source


def move_dir_contents(source, target, overwrite=True):
    file_names = os.listdir(source)
    for file_name in file_names:
        if overwrite:
            # Check if file already exists
            obj = os.path.join(target, file_name)
            remove_object(obj)
        shutil.move(
            os.path.join(source, file_name), target)


def make_dir(target_dir):
    os.makedirs(target_dir, exist_ok=True)


def clean_dir(target_dir):
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    elif os.path.isfile(target_dir):
        raise RuntimeError(
            "The target {} is a file.".format(target_dir))
    os.makedirs(target_dir, exist_ok=True)


def clean_file(target_file):
    if os.path.isfile(target_file):
        os.remove(target_file)
    elif os.path.isdir(target_file):
        raise RuntimeError(
            "The target {} is a directory.".format(target_file))
    # make sure its dir exists
    target_dir = os.path.dirname(target_file)
    os.makedirs(target_dir, exist_ok=True)


def remove_object(target):
    if os.path.isdir(target):
        shutil.rmtree(target)
    elif os.path.isfile(target):
        os.remove(target)


def remove_dir(target_dir):
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    elif os.path.isfile(target_dir):
        raise RuntimeError(
            "The target {} is a file.".format(target_dir))


def remove_file(target_file):
    if os.path.isdir(target_file):
        raise RuntimeError(
            "The target {} is a directory.".format(target_file))
    elif os.path.isfile(target_file):
        os.remove(target_file)
