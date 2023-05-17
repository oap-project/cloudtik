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
# SPDX-License-Identifier: EPL-2.0
#

import os
import torch
import time
import math
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Displays the progress meter during training
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        curr_loss = float(entries[-1].split()[-1][1:-1])
        return curr_loss

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def _fit_simsiam(train_data_loader, model, criterion, optimizer, epoch, precision):
    """
    Main PyTorch Simsiam training loop
    """
    print_freq = 1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')

    progress = ProgressMeter(len(train_data_loader), [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()

    for i, (inputs, _) in enumerate(train_data_loader):
        optimizer.zero_grad()

        data_time.update(time.time() - end)
        inputs[0] = inputs[0].to('cpu')
        inputs[1] = inputs[1].to('cpu')
        with torch.cpu.amp.autocast(enabled=(precision == 'bfloat16')):
            p1, p2, z1, z2 = model(x1=inputs[0], x2=inputs[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), inputs[0].size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            curr_loss = progress.display(i)
    return curr_loss


def _fit_cutpaste(train_data_loader, model, criterion, optimizer, epoch, freeze_resnet, scheduler, precision):
    """
    Main PyTorch CutPaste training loop
    """
    print_freq = 1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_data_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    device = 'cpu'
    if epoch == freeze_resnet:
        print(epoch)
        model.unfreeze()

    end = time.time()
    for i, (data, _) in enumerate(train_data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        xs = [x.to(device) for x in data]
        # zero the parameter gradients
        optimizer.zero_grad()
        xc = torch.cat(xs, axis=0)
        # calculate label
        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))

        with torch.cpu.amp.autocast(enabled=(precision == 'bfloat16')):
            embeds, logits = model(xc)
            loss = criterion(logits, y)

        losses.update(loss.item(), len(data))
        # regulize weights:
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if scheduler is not None:
            scheduler.step(epoch)
        if i % print_freq == 0:
            curr_loss = progress.display(i)
    return curr_loss


def save_checkpoint(state, is_best, filename, loss, checkpoint_dir):
    """
    Custom function to save and rename checkpoints based on the best accuracy model
    """
    if is_best:
        print("Saving a new checkpoint with loss ", loss,
              " at path ", os.path.join(checkpoint_dir, filename))
        torch.save(state, os.path.join(checkpoint_dir, filename))


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """
    Decay the learning rate based on schedule
    """
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def find_threshold(fpr, tpr, thr):
    """
    Compute threshold for calculating accuracy.
    """
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thr))
    return np.round(j_ordered[-1][1], 2)
