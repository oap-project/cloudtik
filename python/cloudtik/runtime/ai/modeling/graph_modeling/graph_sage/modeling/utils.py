"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Chen Haifeng
"""

import argparse
import os

import torch

from cloudtik.runtime.ai.util.utils import clean_file


def existing_directory(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def existing_file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def existing_path(raw_path):
    if not os.path.exists(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory or file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def torch_save(
        obj: object, target_file):
    clean_file(target_file)
    torch.save(obj, target_file)


def df_to_csv(
        df, target_file, index=True):
    clean_file(target_file)
    df.to_csv(target_file, index=index)
