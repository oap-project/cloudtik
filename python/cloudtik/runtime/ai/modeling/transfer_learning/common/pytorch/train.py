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

import argparse

from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.trainer import Trainer, \
    TrainArguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training with PyTorch.')

    parser.add_argument('--objects-path', type=str, required=True,
                        help='The shared data path to load data and model objects.')
    parser.add_argument('--category', type=str, required=True,
                        help='Model category (image_classification|text_classification)')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Total epochs to train the model')
    parser.add_argument('--batch-size', type=int, required=False, default=128,
                        help='Global batch size to distribute data (default: 128)')
    parser.add_argument('--ipex', action='store_true', required=False, default=False,
                        help="Enable IPEX optimization to the model")
    parser.add_argument('--backend', type=str, required=False, default='gloo',
                        help='Type of backend to use (default: gloo)')
    parser.add_argument('--master-addr', type=str, required=False, default='', help="Master node to run this script")
    parser.add_argument('--master-port', type=str, required=False, default='', help='Master port')

    args = parser.parse_args()

    # Load the saved dataset and model objects
    loaded_objects = Trainer.load_objects(
        objects_path=args.objects_path)

    dataset = loaded_objects['dataset']
    train_subset = loaded_objects.get('train_subset', dataset)
    test_subset = loaded_objects.get('test_subset', dataset)
    validation_subset = loaded_objects.get('validation_subset', dataset)
    model = loaded_objects['model']
    loss = loaded_objects['loss']
    optimizer = loaded_objects['optimizer']

    # Launch distributed job
    train_args = TrainArguments(
        category=args.category,
        dataset=train_subset,
        model=model,
        criterion=loss,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ipex=args.ipex
    )

    trainer = Trainer()
    trainer.run(
        train_args, args.master_addr, args.master_port, args.backend)
