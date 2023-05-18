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

import inspect
import os
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch.utils.data import DataLoader as loader
from torchvision.datasets import DatasetFolder
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from cloudtik.runtime.ai.modeling.transfer_learning.image_anomaly_detection.pytorch.simsiam import loader as ssloader
from cloudtik.runtime.ai.modeling.transfer_learning.image_anomaly_detection.pytorch.cutpaste.cutpaste import \
    CutPasteNormal, CutPasteScar,CutPaste3Way, CutPasteUnion, get_cutpaste_transforms
from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.dataset import PyTorchDataset


class AnomalyImageFolder(DatasetFolder):
    """Inherits from DatasetFolder.

    This class overrides the find_classes() and make_dataset() methods of DatasetFolder to support filtering for
    specific defect/class folders and searching the 'train' and 'test' subdirectories when using the optional
    train/test image folder layout.
    """

    def has_valid_file_extension(self, filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        """Checks if a file has a valid extension.

        Args:
            filename (string): path to a file
            extensions (tuple of strings): extensions to consider (lowercase)

        Returns:
            bool: True if the filename ends with one of given extensions
        """
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset and assigns them to 'good' and 'bad'.

        See :class:`DatasetFolder` for details.
        """
        if self._classes_to_find is None:
            classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
            self._defects = [c for c in classes if c != 'good']
        elif isinstance(self._classes_to_find, list):
            classes = self._classes_to_find
        class_to_idx = {cls_name: int(cls_name == 'good') for cls_name in classes}
        return ['bad', 'good'], class_to_idx

    def make_dataset(
            self,
            directory: str,
            class_to_idx: Optional[Dict[str, int]] = None,
            extensions: Optional[Union[str, Tuple[str, ...]]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is optional and will use the logic of the `find_classes` function by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return self.has_valid_file_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            # This is the specific edit that supports nested train/test subdirs for Intel Transfer Learning Tool
            target_dirs = [os.path.join(directory, target_class),
                           os.path.join(directory, 'train', target_class),
                           os.path.join(directory, 'test', target_class)]
            for target_dir in target_dirs:
                if not os.path.isdir(target_dir):
                    continue
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            item = path, class_index
                            instances.append(item)

                            if target_class not in available_classes:
                                available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: " \
                       f"{extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            classes=None
    ):
        self._classes_to_find = classes
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


class PyTorchImageAnomalyDetectionDataset(PyTorchDataset):
    """
    A custom image anomaly detection dataset that can be used with PyTorch models. Note that the
    directory of images is expected to be organized in one of two ways.

    Method 1: With one subfolder named `good` and at least one other folder of defective examples. It does not matter
    what the names of the other folders are or how many there are, as long as there is at least one. All of the images
    in the non-good subfolders will be coded to `bad` and will only be used for validation/testing (not training).

    .. code-block:: text

        dataset_dir
          ├── good
          ├── defective_type_a
          └── defective_type_b

    Method 2: With subfolders named `train` and either `validation` or `test`. The `train` subdirectory should
    contain a folder named `good` with training samples, and the `test`/`validation` subdirectory should contain
    a folder named `good` and at least one other folder of defective examples for validation.

    .. code-block:: text

        dataset_dir
          └── train
              └── good
          └── test
              ├── good
              ├── defective_type_a
              └── defective_type_b

    Args:
        dataset_dir (str): Directory where the data is located. It should contain subdirectories with images for
                           each class.
        dataset_name (str): optional; Name of the dataset. If no dataset name is given, the dataset_dir folder name
                            will be used as the dataset name.
        num_workers (int): optional; Number of processes to use for data loading, default is 56
        shuffle_files (bool): optional; Whether to shuffle the data. Defaults to True.
        defects (list[str]): Specific defects or category names to use for validation (default: None); if None, all
                             subfolders in the dataset directory will be used.

    Raises:
        FileNotFoundError if dataset directory does not exist or if a subdirectory named `good` is not found

    """

    def __init__(self, dataset_dir, dataset_name=None, num_workers=56, shuffle_files=True, defects=None):
        """
        Class constructor
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError("The dataset directory ({}) does not exist".format(dataset_dir))

        # Determine which layout the images are in - category folders or train/test folders
        # The validation_type will be None for the former and "defined_split" for the latter
        if os.path.exists(os.path.join(dataset_dir, 'train')):
            self._validation_type = 'defined_split'
            if not os.path.exists(os.path.join(dataset_dir, 'train', 'good')):
                raise FileNotFoundError("Couldn't find 'good' folder in {}".format(os.path.join(dataset_dir, 'train')))
            if os.path.exists(os.path.join(dataset_dir, 'validation')):
                validation_dir = 'validation'
            elif os.path.exists(os.path.join(dataset_dir, 'test')):
                validation_dir = 'test'
            else:
                raise FileNotFoundError("Found a 'train' directory, but not a 'test' or 'validation' directory.")
        else:
            self._validation_type = None
            if not os.path.exists(os.path.join(dataset_dir, 'good')):
                raise FileNotFoundError("Couldn't find 'good' folder in {}".format(dataset_dir))

        # Inspect and validate defects
        if self._validation_type is None:
            defect_directory = dataset_dir
        elif self._validation_type == 'defined_split':
            defect_directory = os.path.join(dataset_dir, validation_dir)
        classes = [d.name for d in os.scandir(defect_directory) if d.is_dir()]

        if defects:
            # If the defects filter is being used, check that all the folders are present
            complement = set(defects) - set(classes)
            if complement:
                raise ValueError('Some of the defects provided were not found in {}: {}'.format(defect_directory,
                                                                                                complement))
        else:
            # Detect the defects if not provided by user
            defects = [c for c in classes if c != 'good']
        self._defects = defects

        # The dataset name is only used for informational purposes. If one isn't given, use the directory name
        if not dataset_name:
            dataset_name = os.path.basename(dataset_dir)

        PyTorchDataset.__init__(self, dataset_dir, dataset_name, dataset_catalog=None)

        self._info = {
            "name": dataset_name,
            "dataset_dir": dataset_dir
        }
        self._num_workers = num_workers
        self.train_sampler = None
        self._shuffle = self.train_sampler is None
        self._preprocessed = None
        self._train_indices = None
        self._validation_indices = None
        self._test_indices = None

        valid_classes = ['good'] + defects
        self._dataset = AnomalyImageFolder(dataset_dir, classes=valid_classes)

        # For the train/test layout, initialize indices for the train and test subsets
        if self._validation_type == 'defined_split':
            train_img_string = '{}/train/'.format(dataset_dir)
            self._train_indices = [i for i, t in enumerate(self._dataset.imgs) if train_img_string in t[0]]
            self._test_indices = [i for i, t in enumerate(self._dataset.imgs) if train_img_string not in t[0]]
            if self._shuffle:
                random.shuffle(self._train_indices)
                random.shuffle(self._test_indices)

        self._train_subset = None
        self._validation_subset = None
        self._test_subset = None

        self._class_names = self._dataset.classes
        if not self._defects:
            self._defects = self._dataset._defects

        self._train_pct = 1.0
        self._val_pct = 0
        self._test_pct = 0

        self._simsiam_transform = None
        self._cutpaste_transform = None
        self._train_transform = None
        self._validation_transform = None

    @property
    def class_names(self):
        """
        Returns the list of class names
        """
        return self._class_names

    @property
    def defect_names(self):
        """
        Returns the list of class names
        """
        return self._defects

    @property
    def info(self):
        """
        Returns a dictionary of information about the dataset
        """
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}

    @property
    def dataset(self):
        """
        Returns the framework dataset object (torch.utils.data.Dataset)
        """
        return self._dataset

    def shuffle_split(self, train_pct=.75, val_pct=0.25, test_pct=0.0, shuffle_files=True, seed=None):
        """
        Randomly split the good examples into train, validation, and test subsets with a pseudo-random seed option.
        All of the bad examples will be split into validation and test subsets with a similar proportion
        to the val_pct and test_pct arguments.


            Args:
                train_pct (float): default .75, percentage of good examples to use for training
                val_pct (float):  default .25, percentage of good examples to use for validation
                test_pct (float): default 0.0, percentage of good examples to use for testing
                shuffle_files (bool): default True, optionally control whether shuffling occurs
                seed (None or int): default None, can be set for pseudo-randomization

            Raises:
                ValueError if percentage input args are not floats or sum to greater than 1
        """
        if not (isinstance(train_pct, float) and isinstance(val_pct, float) and isinstance(test_pct, float)):
            raise ValueError("Percentage arguments must be floats.")
        if train_pct + val_pct + test_pct > 1.0:
            raise ValueError("Sum of percentage arguments must be less than or equal to 1.")

        good_indices = [i for i, t in enumerate(self._dataset.targets) if t == 1]
        bad_indices = [i for i, t in enumerate(self._dataset.targets) if t == 0]

        good_length = len(good_indices)
        good_train_size = int(train_pct * good_length)
        good_val_size = int(val_pct * good_length)

        # By default 100% of the bad samples will be used in the validation set. But if test_pct is positive,
        # use the ratio of val_pct/test_pct to determine the split of bad samples
        bad_val_pct = 1.0
        if test_pct:
            ratio = val_pct / test_pct
            bad_val_pct = ratio / (ratio + 1)

        bad_length = len(bad_indices)
        bad_val_size = int(bad_val_pct * bad_length)

        generator = torch.Generator().manual_seed(seed) if seed else None
        if shuffle_files:
            random.Random(seed).shuffle(good_indices)
            random.Random(seed).shuffle(bad_indices)
        self._train_indices = good_indices[:good_train_size]
        self._validation_indices = good_indices[good_train_size:good_train_size + good_val_size] + \
            bad_indices[:bad_val_size]
        if test_pct:
            self._test_indices = good_indices[good_train_size + good_val_size:] + \
                bad_indices[bad_val_size:]
        else:
            self._test_indices = None
        self._validation_type = 'shuffle_split'
        if self._preprocessed and 'batch_size' in self._preprocessed:
            self._make_data_loaders(batch_size=self._preprocessed['batch_size'], generator=generator)

    def _make_data_loaders(self, batch_size, generator=None):
        """Make data loaders for the whole dataset and the subsets that have indices defined. Note that this only
        concerns indices, not transforms. The transforms get applied when the dataloaders are used, not created, so
        we need to switch transforms appropriately for train/val subsets when the data is ingested."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if self._dataset:
            # The data_loader for the whole dataset should only have the good samples
            good_indices = [idx for idx, target in enumerate(self._dataset.targets) if target == 1]
            good_samples = torch.utils.data.Subset(self._dataset, good_indices)
            self._data_loader = loader(good_samples, batch_size=batch_size, shuffle=self._shuffle,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator,
                                       pin_memory=True, sampler=self.train_sampler, drop_last=False)
        else:
            self._data_loader = None
        if self._train_indices:
            self._train_loader = loader(self.train_subset, batch_size=batch_size, shuffle=self._shuffle,
                                        num_workers=self._num_workers, worker_init_fn=seed_worker, generator=generator,
                                        pin_memory=True, sampler=self.train_sampler, drop_last=False)
        else:
            self._train_loader = None
        if self._validation_indices or (self._validation_type == 'defined_split' and self._validation_subset):
            self._validation_loader = loader(self.validation_subset, batch_size=batch_size, shuffle=self._shuffle,
                                             num_workers=self._num_workers, worker_init_fn=seed_worker,
                                             generator=generator, pin_memory=True, sampler=self.train_sampler,
                                             drop_last=False)
        else:
            self._validation_loader = None
        if self._test_indices or (self._validation_type == 'defined_split' and self._test_subset):
            self._test_loader = loader(self.test_subset, batch_size=batch_size, shuffle=self._shuffle,
                                       num_workers=self._num_workers, worker_init_fn=seed_worker,
                                       generator=generator, pin_memory=True, sampler=self.train_sampler,
                                       drop_last=False)
        else:
            self._test_loader = None

    def simsiam_transform(self, image_size):
        """
        Perform TwoCropsTransform and GaussianBlur on the dataset for SIMSIAM training.

        Args:
            image_size (int): desired image size

        """
        augmentation = [T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                        T.RandomApply(
                            [T.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8),
                        T.RandomGrayscale(p=0.2),
                        T.RandomApply(
                            [ssloader.GaussianBlur([.1, 2.])], p=0.5),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]

        return ssloader.TwoCropsTransform(T.Compose(augmentation))

    def preprocess(self, image_size=224, batch_size=64, add_aug=None, cutpaste_type='normal', **kwargs):
        """
        Preprocess the dataset to resize, normalize, and batch the images. Apply augmentation
        if specified.

            Args:
                image_size (int or 'variable'): desired square image size (if 'variable', does not alter image size)
                batch_size (int): desired batch size (default 64)
                add_aug (None or list[str]): choice of augmentations ('hflip', 'rotate') to be
                                             applied during training
                cutpaste_type (str): choice of cutpaste variant ('normal', 'scar', '3way', 'union'),
                                     default is 'normal'
                kwargs: optional; additional keyword arguments for Resize and Normalize transforms
            Raises:
                ValueError if the dataset is not defined or has already been processed
        """
        if not (self._dataset):
            raise ValueError("Unable to preprocess, because the dataset hasn't been defined.")

        if self._preprocessed:
            raise ValueError("Data has already been preprocessed: {}".format(self._preprocessed))

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size should be an positive integer")

        if not image_size == 'variable' and not (isinstance(image_size, int) and image_size >= 1):
            raise ValueError("Input image_size must be either a positive int or 'variable'")

        # Get the user-specified keyword arguments
        resize_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(T.Resize).args}
        normalize_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(T.Normalize).args}

        variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar,
                       '3way': CutPaste3Way, 'union': CutPasteUnion}
        variant = variant_map[cutpaste_type]

        def get_transform(image_size, add_aug, train=True):
            """The train argument, if True, will add augmentation transforms, while if False, will add only the
            Resize and Normalize transforms for validation."""
            transforms = []
            if isinstance(image_size, int):
                transforms.append(T.Resize([image_size, image_size], **resize_args))
            if train and add_aug is not None:
                aug_dict = {'hflip': T.RandomHorizontalFlip(),
                            'rotate': T.RandomRotation(0.5)}
                aug_list = ['hflip', 'rotate']
                for option in add_aug:
                    if option not in aug_list:
                        raise ValueError("Unsupported augmentation for PyTorch:{}. \
                        Supported augmentations are {}".format(option, aug_list))
                    transforms.append(aug_dict[option])
            transforms.append(T.ToTensor())
            transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], **normalize_args))

            return T.Compose(transforms)

        self._simsiam_transform = self.simsiam_transform(image_size)
        self._cutpaste_transform = get_cutpaste_transforms(image_size, variant)
        self._train_transform = get_transform(image_size, add_aug, True)
        self._validation_transform = get_transform(image_size, add_aug, False)
        self._preprocessed = {'image_size': image_size, 'batch_size': batch_size}
        self._make_data_loaders(batch_size=batch_size)

    def get_batch(self, subset='all', simsiam=False, cutpaste=False):
        """
        Get a single batch of images and labels from the dataset.

            Args:
                subset (str): default "all", can also be "train", "validation", or "test"
                simsiam (bool): if preprocess() has been previously used on the dataset and this argument is True,
                                the simsiam transform will be applied, otherwise it will not; default False
                cutpaste (bool): if preprocess() has been previously used on the dataset and this argument is True,
                                the cutpaste transform will be applied, otherwise it will not; default False

            Returns:
                (examples, labels)

            Raises:
                ValueError if the dataset is not defined yet or the given subset is not valid
        """
        if simsiam:
            # SimSiam transform can be manually requested with any subset
            self._dataset.transform = self._simsiam_transform
        elif cutpaste:
            # CutPaste transform can be manually requested with any subset
            self._dataset.transform = self._cutpaste_transform
        elif subset in ['all', 'train']:
            # For "train"/"all" subsets, if simsiam and cutpaste are False,
            # the train transform (including augmentation) is applied
            self._dataset.transform = self._train_transform
        else:
            # For "validation"/"test" subsets, if simsiam and cutpaste are False,
            # the validation transform (excluding augmentation) is applied
            self._dataset.transform = self._validation_transform

        if subset == 'all' and self._dataset is not None:
            return next(iter(self._data_loader))
        elif subset == 'train' and self._train_loader is not None:
            return next(iter(self._train_loader))
        elif subset == 'validation' and self._validation_loader is not None:
            return next(iter(self._validation_loader))
        elif subset == 'test' and self._test_loader is not None:
            return next(iter(self._test_loader))
        else:
            raise ValueError("Unable to return a batch, because the dataset or subset hasn't been defined.")
