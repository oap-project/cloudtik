#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
# SPDX-License-Identifier: Apache-2.0
#

import os
from pydoc import locate
import tarfile
import zipfile

from .sources import DatasetSource
from .. import utils


class DataDownloader:
    """
    A unified dataset downloader class.

    Can download from TensorFlow Datasets, Torchvision, Hugging Face, and generic web URLs. If initialized for a
    dataset catalog, the download method will return a dataset object of type tensorflow.data.Dataset,
    torch.utils.data.Dataset, or datasets.arrow_dataset.Dataset. If initialized for a web URL that is a zipfile or a
    tarfile, the file will be extracted and the path, or list of paths, to the extracted contents will be returned.
    """
    def __init__(self, dataset_name, dataset_dir, source=None, url=None, **kwargs):
        """
        Class constructor for a DataDownloader.

            Args:
                dataset_name (str): Name of the dataset
                dataset_dir (str): Local destination directory of dataset
                source (str, optional): The source to download the dataset from; options are 'tensorflow_datasets',
                    'torchvision', 'hugging_face', and None which will result in a GENERIC type dataset which expects
                    an accompanying url input
                url (str, optional): If downloading from the web, provide the URL location
                kwargs (optional): Some catalogs accept additional keyword arguments when downloading

            raises:
                ValueError if both catalog and url are omitted or if both are provided

        """
        if source is None and url is None:
            raise ValueError("Must provide either a source or url as the source.")
        if source is not None and url is not None:
            raise ValueError(
                "Only one of source or url should be provided. Found {} and {}.".format(source, url))

        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        self._dataset_name = dataset_name
        self._dataset_dir = dataset_dir
        self._source = DatasetSource.from_str(source)
        self._url = url
        self._args = kwargs

    def download(self, split='train'):
        """
        Download the dataset

            Args:
                split (str): desired split, optional

            Returns:
                tensorflow.data.Dataset, torch.utils.data.Dataset, datasets.arrow_dataset.Dataset, str, or list[str]

        """
        if self._source == DatasetSource.TENSORFLOW_DATASETS:
            import tensorflow_datasets as tfds
            if isinstance(split, str):
                split = [split]
            os.environ['NO_GCE_CHECK'] = 'true'
            return tfds.load(self._dataset_name,
                             data_dir=self._dataset_dir,
                             split=split,
                             **self._args)

        elif self._source == DatasetSource.TORCHVISION:
            dataset_class = locate('torchvision.datasets.{}'.format(self._dataset_name))
            try:
                return dataset_class(self._dataset_dir, download=True, split=split)
            except TypeError:
                return dataset_class(self._dataset_dir, download=True, train=split == 'train')

        elif self._source == DatasetSource.HUGGING_FACE:
            from datasets import load_dataset
            if 'subset' in self._args:
                return load_dataset(self._dataset_name, self._args['subset'], split=split, cache_dir=self._dataset_dir)
            else:
                return load_dataset(self._dataset_name, split=split, cache_dir=self._dataset_dir)

        elif self._source == DatasetSource.GENERIC:
            file_path = utils.download_file(self._url, self._dataset_dir)
            if os.path.isfile(file_path):
                if tarfile.is_tarfile(file_path):
                    contents = utils.extract_tar_file(file_path, self._dataset_dir)
                elif zipfile.is_zipfile(file_path):
                    contents = utils.extract_zip_file(file_path, self._dataset_dir)
                else:
                    return file_path

                # Contents are a list of top-level extracted members
                # Convert to absolute paths and return a single string if length is 1
                if len(contents) > 1:
                    return [os.path.join(self._dataset_dir, i) for i in contents]
                else:
                    return os.path.join(self._dataset_dir, contents[0])

            else:
                raise FileNotFoundError("Unable to find the downloaded file at:", file_path)
