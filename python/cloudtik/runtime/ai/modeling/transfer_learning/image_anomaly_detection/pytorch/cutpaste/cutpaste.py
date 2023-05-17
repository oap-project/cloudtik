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

import random
import math
from torchvision import transforms
import torch


def get_cutpaste_transforms(size, cutpaste_type):
    """
    Apply cutpaste transforms

    Args:
        cutpaste_type : variant map of the type of cutpaste used for training
    """
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size, size)))
    train_transform.transforms.append(cutpaste_type(transform=after_cutpaste_transform))
    return train_transform


class CutPaste(object):
    """
    Base class for cutpaste variants with common operations
    """
    def __init__(self, colorJitter=0.1, transform=None):
        """
        Class constructor
        """
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=colorJitter,
                                                      contrast=colorJitter,
                                                      saturation=colorJitter,
                                                      hue=colorJitter)

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img


class CutPasteNormal(CutPaste):
    """
    Randomly copy one patch from the image and paste it somewhere else.

    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area
                           to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between
                              aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwargs):
        """
        Class constructor
        """
        super(CutPasteNormal, self).__init__(**kwargs)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):

        h = img.size[0]
        w = img.size[1]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)

        return super().__call__(img, augmented)


class CutPasteScar(CutPaste):
    """
    Randomly copy a scar-like (long-thin) rectangular patch
    from the image and paste it somewhere else.

    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2, 16], height=[10, 25], rotation=[-45, 45], **kwargs):
        """
        Class constructor
        """
        super(CutPasteScar, self).__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation

    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]

        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)

        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)

        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)

        # paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")

        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)

        return super().__call__(img, augmented)


class CutPasteUnion(object):
    """
    Pick a random selection between CutPasteNormal and CutPasteScar
    """
    def __init__(self, **kwargs):
        """
        Class constructor
        """
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)

    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)


class CutPaste3Way(object):
    """
    Generate CutPasteScar and CutPasteUnion images
    """
    def __init__(self, **kwargs):
        """
        Class constructor
        """
        self.normal = CutPasteNormal(**kwargs)
        self.scar = CutPasteScar(**kwargs)

    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)

        return org, cutpaste_normal, cutpaste_scar
