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

import json
import os
import urllib.request
import re
import shutil
import tarfile
from zipfile import ZipFile


def read_json_file(json_file_path):
    """
    Reads a json file an returns a dictionary representing the file contents

    :param json_file_path: Path to the json file
    :return: Dictionary
    """

    if not os.path.isfile(json_file_path):
        raise FileNotFoundError("The json file {} does not exist".format(json_file_path))

    with open(json_file_path, "r") as f:
        data = json.load(f)

    return data


def download_file(download_url, destination_directory):
    """
    Downloads a file using the specified url to the destination directory. Returns the
    path to the downloaded file.
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    destination_file_path = os.path.join(destination_directory, os.path.basename(download_url))

    print("Downloading {} to {}".format(download_url, destination_directory))
    with urllib.request.urlopen(download_url) as response, open(destination_file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    return destination_file_path


def extract_tar_file(tar_file_path, destination_directory):
    """
    Extracts a tar file on the local file system to the destination directory
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    print("Extracting {} to {}".format(tar_file_path, destination_directory))
    with tarfile.open(tar_file_path) as t:
        t.extractall(path=destination_directory)


def extract_zip_file(zip_file_path, destination_directory):
    """
    Extracts a zip file on the local file system to the destination directory
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    print("Extracting {} to {}".format(zip_file_path, destination_directory))
    with ZipFile(zip_file_path, "r") as zipfile:
        zipfile.extractall(path=destination_directory)


def download_and_extract_tar_file(tar_file_url, destination_directory):
    """
    Downloads a tar file using the specified URL to the destination directory, then extracts
    the tar file to the destination directory.
    """
    local_tar_path = download_file(tar_file_url, destination_directory)

    if os.path.isfile(local_tar_path):
        extract_tar_file(local_tar_path, destination_directory)
    else:
        raise FileNotFoundError("Unable to find the downloaded tar file at:", local_tar_path)


def download_and_extract_zip_file(zip_file_url, destination_directory):
    """
    Downloads a tar file using the specified URL to the destination directory, then extracts
    the zip file to the destination directory.
    """
    local_zip_path = download_file(zip_file_url, destination_directory)

    if os.path.isfile(local_zip_path):
        extract_zip_file(local_zip_path, destination_directory)
    else:
        raise FileNotFoundError("Unable to find the downloaded zip file at:", local_zip_path)


def verify_directory(directory_path, require_directory_exists=False):
    """
    Verifies that the input parameter is a string and that it's not already a file. If require_directory_exists is
    True, and the directory does not exist, a NotADirectoryError is raised. Otherwise, if the directory does not
    exist it will be created.
    """

    if not isinstance(directory_path, str):
        raise TypeError("The directory path should be a str, but was a {}".format(type(directory_path)))

    if require_directory_exists and not os.path.isdir(directory_path):
        raise NotADirectoryError("The directory does not exist at:", directory_path)

    if os.path.isfile(directory_path):
        raise FileExistsError("Unable to use directory path {} because it already exists as "
                              "file".format(directory_path))

    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def validate_model_name(model_name):
    """
    Verifies that the input parameter is a string. If the input parameter is indeed a string, a regular expression
    is used to clean the string of white spaces and any non-alphanumeric characters (besides dashes and underscores).
    Any matches will be replaced with an underscore.
    """
    if not isinstance(model_name, str):
        raise TypeError("The model name should be a str, but was a {}".format(type(model_name)))
    else:
        model_name = model_name.strip()
        model_name = " ".join(model_name.split())
        model_name = re.sub('[^a-zA-Z\d_-]', '_', model_name)  # noqa: W605
        return model_name


def get_framework_name(framework):
    if "pytorch" == framework:
        return "PyTorch"
    elif "tensorflow" == framework:
        return "Tensorflow"
    raise ValueError("Unknown framework type.")


def get_category_name(category):
    parts = category.split('_')
    return ''.join(part.title() for part in parts)
