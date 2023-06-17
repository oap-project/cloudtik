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
            if os.path.isdir(obj):
                print('Directory {} exists in the {}. Remove.'.format(file_name, target))
                shutil.rmtree(obj)
            elif os.path.isfile(obj):
                print('File {} exists in the {}. Remove.'.format(file_name, target))
                os.remove(obj)
        shutil.move(
            os.path.join(source, file_name), target)
