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

import sys
import torch.nn as nn
from torchvision.models import resnet18, resnet50


class ProjectionNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True,
                 head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ProjectionNet, self).__init__()
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained)
            last_layer = 512
        elif model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            last_layer = 2048
        else:
            sys.exit("ERROR, only supported models are resnet18 and resnet50")

        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # last layer without activation
        head = nn.Sequential(*sequential_layers)
        self.model.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.model(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        # freeze full resnet18
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze head:
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
