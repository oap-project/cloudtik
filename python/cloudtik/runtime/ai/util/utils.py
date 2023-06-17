from pathlib import Path

import json
import yaml


def load_config_from_file(config_file, config=None):
    if config_file.endswith(".json"):
        args_in_file = json.loads(Path(config_file).read_text())
    else:
        # default a yaml file
        with open(config_file, "r") as f:
            args_in_file = yaml.safe_load(f)

    if config is not None:
        for key in args_in_file:
            setattr(config, key, args_in_file[key])
        return config

    return args_in_file
