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
# SPDX-License-Identifier: Apache-2.0
#
import datetime
import inspect
import os
import time
import yaml
import dill
import numpy as np
from requests.adapters import ProxyError

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    get_scheduler,
    set_seed
)

from datasets.arrow_dataset import Dataset

from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.model import PyTorchModel
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.pytorch.hugging_face.text_classification_dataset \
    import HuggingFaceTextClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.pytorch.text_classification_dataset \
    import PyTorchTextClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.text_classification_dataset \
    import TextClassificationDataset
from cloudtik.runtime.ai.modeling.transfer_learning.text_classification.text_classification_model \
    import TextClassificationModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.downloader.models \
    import ModelDownloader
from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.hugging_face.model \
    import HuggingFaceModel
from cloudtik.runtime.ai.modeling.transfer_learning.common.utils \
    import read_json_file, validate_model_name, verify_directory


class HuggingFaceTextClassificationModel(TextClassificationModel, HuggingFaceModel):
    """
    Class to represent a Hugging Face pretrained model that can be used for multi-class text classification
    fine tuning.
    """

    def __init__(self, model_name: str, model=None,
                 optimizer=None, loss=None, **kwargs):
        this_dir = os.path.dirname(__file__)
        model_map = read_json_file(os.path.join(
            this_dir, "text_classification_models.json"))

        # extra properties that will become configurable in the future
        self._model_name = model_name
        self._dropout_layer_rate = 0.1
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._lr_scheduler = None
        self._generate_checkpoints = True
        self._tokenizer = None
        self._classification_layer = model_map[model_name]["classification_layer"]

        TextClassificationModel.__init__(self, model_name, self._dropout_layer_rate)
        HuggingFaceModel.__init__(self, model_name)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        self._optimizer_class = optimizer if optimizer else torch.optim.AdamW
        self._opt_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else torch.nn.CrossEntropyLoss
        self._loss_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        # model definition
        self.hub_name = model_map[self._model_name]["hub_name"]
        self._model = None
        self._num_classes = None
        self._trainer = None
        self._history = None

    def save_objects(self, dataset, output_dir):
        """
        Helper function to export dataset and model objects to disk for distributed job

        Args:
            output_dir (str): Path to a directory where the dataset and model objects are saved.
                Default file name for saving the objects is "hf_saved_objects.obj"
            dataset (HuggingFaceTextClassificationDataset): Dataset object to save. It must be an object of
                HuggingFaceTextClassificationDataset so that the dataset info, train, test, and validation
                subsets can be accessed.
        """

        objects_to_save = {
            "dataset": dataset.dataset,
            "info": dataset.info,
            "train_subset": dataset.train_subset,
            "test_subset": dataset.test_subset,
            "validation_subset": dataset.validation_subset,
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": self._loss
        }
        now = datetime.datetime.now()
        filename = f"torch_objects_{now:%Y-%m-%d_%H-%M-%S}.obj"
        objects_path = os.path.join(output_dir, filename)
        torch.save(objects_to_save, objects_path)
        return objects_path

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _fit(self, output_dir, dataset, epochs, do_eval, early_stopping, lr_decay):
        train_data_loader = None
        validation_data_loader = None

        if isinstance(dataset, HuggingFaceTextClassificationDataset) or \
                isinstance(dataset, PyTorchTextClassificationDataset):
            if not dataset._preprocessed:
                raise ValueError("dataset is not preprocessed yet")
            self._tokenizer = dataset._tokenizer

            # Get the data loader objects
            train_data_loader = dataset.train_loader
            validation_data_loader = dataset.validation_loader
            train_data_length = len(dataset.train_subset)
        elif isinstance(dataset, Dataset):
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Create new data loader objects
            train_data_loader = DataLoader(dataset, batch_size=16)
            validation_data_loader = DataLoader(dataset, batch_size=16)
            train_data_length = len(dataset)
        else:
            raise ValueError("Invalid dataset type: {}".format(type(dataset)))

        # For early stopping, if enabled
        patience = 10
        trigger_time = 0
        last_loss = 1.0

        num_training_steps = epochs * train_data_length
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self._optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Training loop
        since = time.time()
        self._model.to(self._device)
        self._history = {}
        self._model.train()
        # Training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            # Training phase
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data_batch in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(self._device) for k, v in data_batch.items()
                          if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                labels = data_batch['label']

                # zero the parameter gradients
                self._optimizer.zero_grad()

                # Forward pass
                outputs = self._model(**inputs)
                loss = self._loss(outputs.logits, labels)

                # Backward pass
                loss.backward()
                self._optimizer.step()
                lr_scheduler.step()

                # Statistics
                predictions = torch.argmax(outputs.logits, dim=-1)

                running_loss += loss.item()
                running_corrects += torch.sum(predictions == labels).item()

            # At the epoch end
            train_epoch_loss = running_loss / train_data_length
            train_epoch_acc = running_corrects / train_data_length

            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                eval_epoch_loss, eval_epoch_acc = self.evaluate(validation_data_loader)

                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

                if lr_decay:
                    lr = lr_scheduler.optimizer.param_groups[0]['lr']
                    self._update_history('LR', lr)
                    loss_acc_output += f' - LR: {lr:.4f}'
                    lr_scheduler.step(eval_epoch_loss)

                # Put the model back to train mode
                self._model.train()

            if early_stopping:
                if eval_epoch_loss >= last_loss:
                    trigger_time += 1

                    if trigger_time >= patience:
                        # Stop Early
                        print("Early stopping has been triggered after " + str(epoch) + " epochs.")
                        break
                else:
                    trigger_time = 0

                last_loss = eval_epoch_loss

            print(loss_acc_output)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if self._generate_checkpoints:
            valid_model_name = validate_model_name(self.model_name)
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
            verify_directory(checkpoint_dir)
            try:
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
            except KeyError:
                # Calling state_dict() on an IPEX optimizer calls into the torch optimizer's __setstate__ method
                # which in PyTorch 1.12 assumes that the first state value will always have a 'step' key
                state_values = list(self._optimizer.state.values())
                if 'step' not in state_values[0].keys():
                    state_values[0]['step'] = torch.tensor([])
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

    def _fit_distributed(
            self, nnodes, nproc_per_node, hosts, hostfile,
            epochs, batch_size, ipex_optimize, objects_path):
        PyTorchModel.fit_distributed(
            nnodes, nproc_per_node, hosts, hostfile,
            epochs, batch_size, ipex_optimize,
            objects_path, category="text_classification"
        )

    def train(
            self,
            dataset,
            output_dir: str,
            epochs: int = 1,
            initial_checkpoints=None,
            do_eval: bool = True,
            early_stopping: bool = False,
            lr_decay: bool = True,
            seed: int = None,
            learning_rate: float = 1e-5,
            extra_layers: list = None,
            device: str = "cpu",
            ipex_optimize: bool = False,
            use_trainer: bool = False,
            force_download: bool = False,
            distributed=False,
            nnodes=1,
            nproc_per_node=1,
            hosts=None,
            hostfile=None,
            shared_dir=None,
            **kwargs
    ):
        """
        Trains the model using the specified text classification dataset.

        Args:
            dataset (TextClassificationDataset/datasets.arrow_dataset.Dataset): The dataset to use for training.
                If a train subset has been defined, that subset will be used to fit the model. Otherwise, the
                entire non-partitioned dataset will be used.
            output_dir (str): A writeable output directory to write checkpoint files during training
            epochs (int): The number of training epochs [default: 1]
            initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                latest checkpoint will be used.
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                at the end of each epoch.
            early_stopping (bool): Enable early stopping if convergence is reached while training
                at the end of each epoch.
            learning_rate (float): Learning rate for the model to train. Defaults to 1e-5
            lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                is applied at the end of each epoch.
            seed (int): Optionally set a seed for reproducibility.
            extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                layer. This can help increase accuracy when fine-tuning a PyTorch model.
                The input should be a list of integers representing the number and size of the layers,
                for example [1024, 512] will insert two dense layers, the first with 1024 neurons and the
                second with 512 neurons.
            device (str): Device to train the model. Defaults to "cpu"
            ipex_optimize (bool): Optimize the model using Intel® Extension for PyTorch. Defaults to False
            use_trainer (bool): If use_trainer is True, then the model training is done using the Hugging Face Trainer
                and if use_trainer is False, the model training is done using native PyTorch training loop
            force_download (bool): Downloads the model with default parameters. Defaults to False.
            distributed (bool): Boolean flag to use distributed training. Defaults to False.
            nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
            nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
            to 1.
            hosts (str): hosts list for distributed training. Defaults to None.
            hostfile (str): Name of the hostfile for distributed training. Defaults to None.
            shared_dir (str): The shared data dir for distributed training.

        Returns:
            Dictionary containing the model training history

        Raises:
            TypeError if the dataset specified is not a TextClassificationDataset/datasets.arrow_dataset.Dataset
            ValueError if the given dataset has not been preprocessed yet

        """
        self._check_train_inputs(output_dir, dataset, TextClassificationDataset,
                                 extra_layers, epochs)

        if not self._model:
            self._num_classes = len(dataset.class_names)
            downloader = ModelDownloader(self.hub_name, model_dir=None, source='hugging_face',
                                         num_labels=self._num_classes, force_download=force_download)
            try:
                self._model = downloader.download()
            except ProxyError:
                print('Max retries reached. Sleeping for 10 sec...')
                time.sleep(10)
                self._model = downloader.download()

        if not self._optimizer:
            self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

        self._device = device
        self.train_data_loader = None
        self.validation_data_loader = None

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if extra_layers:
            classifier = getattr(self._model, self._classification_layer[0])
            num_features = classifier.in_features
            setattr(self._model, self._classification_layer[0], torch.nn.Sequential())
            classifier = getattr(self._model, self._classification_layer[0])
            for layer in extra_layers:
                classifier.append(torch.nn.Linear(num_features, layer))
                classifier.append(torch.nn.ReLU(inplace=True))
                num_features = layer
            classifier.append(torch.nn.Linear(num_features, self._num_classes))

        # Initialize the optimizer class and create a learning rate scheduler
        self._optimizer = self._optimizer_class(
            self._model.parameters(), lr=learning_rate, **self._opt_args)

        if seed is not None:
            set_seed(seed)

        if use_trainer:
            if distributed:
                raise ValueError("Distributed training with Trainer is not implemented yet")
            training_args = TrainingArguments(
                output_dir=output_dir,
                do_eval=do_eval,
                do_train=True,
                no_cuda=True,
                overwrite_output_dir=True,
                per_device_train_batch_size=dataset.info['preprocessing_info']['batch_size'],
                evaluation_strategy="epoch",
                num_train_epochs=epochs,
                max_steps=75,
            )

            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.argmax(preds, axis=1)
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

            # Initialize our Trainer
            self._tokenizer = dataset._tokenizer
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset.train_subset,
                eval_dataset=dataset.validation_subset,
                compute_metrics=compute_metrics,
                tokenizer=self._tokenizer
            )

            self._trainer.train()
            if do_eval:
                self._history = self._trainer.evaluate()
                print("Val Acc: {:.5f}".format(self._history.get("eval_accuracy")))
        elif distributed:
            objects_path = self.save_objects(dataset, shared_dir)
            self._fit_distributed(
                nnodes, nproc_per_node, hosts, hostfile,
                epochs, dataset._preprocessed["batch_size"], ipex_optimize,
                objects_path)
        else:
            self._trainer = None
            self._model.train()
            if ipex_optimize:
                import intel_extension_for_pytorch as ipex
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)
            # Call the _fit method to train the model with native PyTorch API
            self._fit(output_dir, dataset, epochs, do_eval, early_stopping, lr_decay)

        return self._history

    def evaluate(self, dataset_or_dataloader=None):
        """
           Evaulates the model on the given dataset (or) dataloader. If Hugging Face Trainer object was used to
           train the model, it evaluates on the 'eval_dataset' given in the Trainer arguments

           Args:
               dataset_or_dataloader (datasets.arrow_dataset.Dataset/DataLoader/TextClassificationDataset): The
                    dataset/dataloader to use for evaluation.

           Returns:
               Tuple with loss and accuracy metrics

           Raises:
               TypeError if the dataset specified is not a datasets.arrow_dataset.Dataset (or) a
                    TextClassificationDataset (or) a DataLoader
        """
        if self._trainer:
            results = self._trainer.evaluate()
            validation_loss = None
            validation_accuracy = results.get("eval_accuracy")
            print("Val Acc: {:.5f}".format(validation_accuracy))
        else:
            if isinstance(dataset_or_dataloader, Dataset):
                dataloader = DataLoader(dataset_or_dataloader, batch_size=16)
                validation_data_length = len(dataset_or_dataloader)
            elif isinstance(dataset_or_dataloader, DataLoader):
                dataloader = dataset_or_dataloader
                validation_data_length = len(dataloader) * dataloader.batch_size
            elif isinstance(dataset_or_dataloader, HuggingFaceTextClassificationDataset) or \
                    isinstance(dataset_or_dataloader, PyTorchTextClassificationDataset):
                dataloader = dataset_or_dataloader.validation_loader
                validation_data_length = len(dataset_or_dataloader)
            else:
                raise TypeError("Invalid dataset/dataloader: {}".format(dataset_or_dataloader))

            if not self._model:
                # The model hasn't been trained yet, use the original transformers model
                self._num_classes = len(dataset_or_dataloader.class_names)
                downloader = ModelDownloader(self.hub_name, source='hugging_face', model_dir=None,
                                             num_labels=self._num_classes)
                self._model = downloader.download()

            # Do the evaluation
            device = torch.device(self._device)
            self._model = self._model.to(device)

            self._model.eval()
            running_loss = 0.0
            running_corrects = 0

            for data_batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(device) for k, v in data_batch.items()
                          if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                labels = data_batch['label'].to(device)

                outputs = self._model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                loss = self._loss(outputs.logits, labels)

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(predictions == labels).item()

            if validation_data_length == 0:
                validation_loss, validation_accuracy = 0.0, 0.0
            else:
                validation_loss = running_loss / validation_data_length
                validation_accuracy = running_corrects / validation_data_length

        return (validation_loss, validation_accuracy)

    def predict(self, input_samples, return_raw=False):
        """
           Generates predictions for the specified input samples.

           Args:
               input_samples (str, list, encoded dict, TextClassificationDataset):
                    Input samples to use to predict.
               return_raw (Bool):
                    Option to return the HF SequenceClassifierOutput object containing the
                    logits Torch Tensor, if set to True.

           Returns:
               Torch Tensor of scores or HF SequenceClassifierOutput if return_raw is set to True.

           Raises:
               NotImplementedError if the given input_samples is of type DataLoader
        """
        encoded_input = None

        # If 'input_samples' is a single text string or a list of text strings
        if isinstance(input_samples, str) or isinstance(input_samples, list):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        # If 'input_samples' is an encoded input dict
        elif isinstance(input_samples, dict):
            # Requires at least 'input_ids' key and any other mentioned below
            required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
            encoded_input = {k: v for k, v in input_samples.items() if k in required_keys}
        # If 'input_samples' is of type HuggingFaceTextClassificationDataset
        elif isinstance(input_samples, HuggingFaceTextClassificationDataset) or\
                isinstance(input_samples, PyTorchTextClassificationDataset):
            if input_samples._preprocessed:
                encoded_input = {
                    'input_ids': input_samples['input_ids'],
                    'attention_mask': input_samples['attention_mask'],
                    'token_type_ids': input_samples['token_type_ids']
                }
        # If the 'input_samples' are already pre-processed, then it will be a Dataset object
        elif isinstance(input_samples, Dataset):
            encoded_input = {
                'input_ids': input_samples['input_ids'],
                'attention_mask': input_samples['attention_mask'],
                'token_type_ids': input_samples['token_type_ids']
            }
        # if 'input_samples' is a DataLoader object
        elif isinstance(input_samples, DataLoader):
            raise NotImplementedError("Prediction using Dataloader hasn't been implmented yet. \
                                Use raw text or Dataset as input!")

        output = self._model(**encoded_input)
        if return_raw:
            return output

        _, predictions = torch.max(output.logits, dim=1)
        return predictions

    def export(self, output_dir: str):
        """
        Saves the model to the given output_dir directory.

        Args:
            output_dir (str): Path to save the model.
        """
        if self._model:
            verify_directory(output_dir)
            valid_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, valid_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")
            verify_directory(saved_model_dir)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
            model_copy = dill.dumps(self._model.module if hasattr(self._model, 'module') else self._model)
            torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")

    def load_from_directory(self, model_dir: str, num_classes: int):
        """
        Loads a saved pytorch model from the given model_dir directory

        Args:
            model_dir(str): Path to the saved model directory
            num_classes(int): Number of class labels
        """

        verify_directory(model_dir, require_directory_exists=True)
        model_copy = torch.load(os.path.join(model_dir, 'model.pt'))
        self._model = dill.loads(model_copy)
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

    def write_inc_config_file(self, config_file_path, dataset, batch_size, overwrite=False,
                              resize_interpolation='bicubic', accuracy_criterion_relative=0.01, exit_policy_timeout=0,
                              exit_policy_max_trials=50, tuning_random_seed=9527,
                              tuning_workspace=''):
        """
        Writes an INC compatible config file to the specified path usings args from the specified dataset and
        parameters.

        Args:
            config_file_path (str): Destination path on where to write the .yaml config file.
            dataset (Dataset): A dataset object
            batch_size (int): Batch size to use for quantization and evaluation
            overwrite (bool): Specify whether or not to overwrite the config_file_path, if it already exists
                              (default: False)
            resize_interpolation (str): Interpolation type. Select from: 'bilinear', 'nearest', 'bicubic'
                                        (default: bicubic)
            accuracy_criterion_relative (float): Relative accuracy loss (default: 0.01, which is 1%)
            exit_policy_timeout (int): Tuning timeout in seconds (default: 0). Tuning processing finishes when the
                                       timeout or max_trials is reached. A tuning timeout of 0 means that the tuning
                                       phase stops when the accuracy criterion is met.
            exit_policy_max_trials (int): Maximum number of tuning trials (default: 50). Tuning processing finishes when
                                          the timeout or or max_trials is reached.
            tuning_random_seed (int): Random seed for deterministic tuning (default: 9527).
            tuning_workspace (dir): Path the INC nc_workspace folder. If the string is empty and the OUTPUT_DIR env var
                                    is set, that output directory will be used. If the string is empty and the
                                    OUTPUT_DIR env var is not set, the default INC nc_workspace location will be used.
        Returns:
            None
        Raises:
            FileExistsError if the config file already exists and overwrite is set to False.
            ValueError if the parameters are not within the expected values
            NotImplementedError if the dataset type is not HFCustomImageClassificationDataset.
        """
        if os.path.isfile(config_file_path) and not overwrite:
            raise FileExistsError('A file already exists at: {}. Provide a new file path or set overwrite=True',
                                  config_file_path)

        # They don't have a PyTorch Dataset option, so for now, we only support custom datasets for quantization

        if not isinstance(dataset, PyTorchTextClassificationDataset) or \
                dataset.__class__ is not PyTorchTextClassificationDataset:
            raise NotImplementedError('quantization has only been implemented for huggingface text classification '
                                      'models with custom datasets')

        if batch_size and not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Invalid value for batch size ({}). Expected a positive integer.'.format(batch_size))

        if resize_interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Invalid value for resize interpolation ({}). Expected one of the following values: '
                             'bilinear, nearest, bicubic'.format(resize_interpolation))

        if accuracy_criterion_relative and not isinstance(accuracy_criterion_relative, float) or \
                not (0.0 <= accuracy_criterion_relative <= 1.0):
            raise ValueError('Invalid value for the accuracy criterion ({}). Expected a float value between 0.0 '
                             'and 1.0'.format(accuracy_criterion_relative))

        if exit_policy_timeout and not isinstance(exit_policy_timeout, int) or exit_policy_timeout < 0:
            raise ValueError('Invalid value for the exit policy timeout ({}). Expected a positive integer or 0.'.
                             format(exit_policy_timeout))

        if exit_policy_max_trials and not isinstance(exit_policy_max_trials, int) or exit_policy_max_trials < 1:
            raise ValueError('Invalid value for max trials ({}). Expected an integer greater than 0.'.
                             format(exit_policy_timeout))

        if tuning_random_seed and not isinstance(tuning_random_seed, int) or tuning_random_seed < 0:
            raise ValueError('Invalid value for tuning random seed ({}). Expected a positive integer.'.
                             format(tuning_random_seed))

        if not isinstance(tuning_workspace, str):
            raise ValueError('Invalid value for the nc_workspace directory. Expected a string.')

        # Get the Intel Neural Compressor template
        config_template = TextClassificationModel.get_inc_config_template_dict(self)

        # Collect the different data loaders into a list, so that we can update them all the with the data transforms
        dataloader_configs = []

        # If tuning_workspace is undefined, use the OUTPUT_DIR, if the env var exists
        if not tuning_workspace:
            output_dir_env_var = os.getenv('OUTPUT_DIR', '')

            if output_dir_env_var:
                tuning_workspace = os.path.join(output_dir_env_var, 'nc_workspace')

        print("tuning_workspace:", tuning_workspace)

        if "quantization" in config_template.keys() and "calibration" in config_template["quantization"].keys() \
                and "dataloader" in config_template["quantization"]["calibration"].keys():
            dataloader_configs.append(config_template["quantization"]["calibration"]["dataloader"])
            print("DATALOADER CONFIGS")
            print(dataloader_configs)

        if "evaluation" in config_template.keys():
            if "accuracy" in config_template["evaluation"].keys() and \
                    "dataloader" in config_template["evaluation"]["accuracy"].keys():
                dataloader_configs.append(config_template["evaluation"]["accuracy"]["dataloader"])
            if "performance" in config_template["evaluation"].keys() and \
                    "dataloader" in config_template["evaluation"]["performance"].keys():
                dataloader_configs.append(config_template["evaluation"]["performance"]["dataloader"])

        config_template["quantization"]["approach"] = "post_training_dynamic_quant"

        # Update the data loader configs
        for dataloader_config in dataloader_configs:
            # Update dataset directory for the custom dataset
            if "dataset" in dataloader_config.keys() and "bert" in dataloader_config["dataset"].keys():
                # These cause errors when trying to benchmark
                dataloader_config["dataset"]["bert"]["root"] = dataset.dataset_dir
                dataloader_config["dataset"]["bert"]["label_file"] = dataset.dataset_dir

            dataloader_config["batch_size"] = batch_size

        if "tuning" in config_template.keys():
            config_template["tuning"]["accuracy_criterion"]["relative"] = accuracy_criterion_relative

            if exit_policy_timeout is None:
                config_template["tuning"]["exit_policy"].pop('timeout', None)
            else:
                config_template["tuning"]["exit_policy"]["timeout"] = exit_policy_timeout

            if exit_policy_max_trials is None:
                config_template["tuning"]["exit_policy"].pop('max_trials', None)
            else:
                config_template["tuning"]["exit_policy"]["max_trials"] = exit_policy_max_trials

            if tuning_random_seed is None:
                config_template["tuning"].pop('random_seed', None)
            else:
                config_template["tuning"]["random_seed"] = tuning_random_seed

            if tuning_workspace:
                if "workspace" not in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"] = {}

                config_template["tuning"]["workspace"]["path"] = tuning_workspace
            else:
                # No tuning_workspace is defined, so remove it from the config
                if "workspace" in config_template["tuning"].keys():
                    config_template["tuning"]["workspace"].pop("path", None)

                    if len(config_template["tuning"]["workspace"].keys()) == 0:
                        config_template["tuning"].pop("workspace", None)

        # Create the directory where the file will be written, if it doesn't already exist
        if not os.path.exists(os.path.dirname(config_file_path)):
            os.makedirs(os.path.dirname(config_file_path))

        # Write the config file
        with open(config_file_path, "w") as config_file:
            yaml.dump(config_template, config_file, sort_keys=False)

    def quantize(self, saved_model_dir, output_dir, inc_config_path):
        """
        Performs post training quantization using the Intel Neural Compressor on the model from the saved_model_dir
        using the specified config file. The quantized model is written to the output directory.

        Args:
            saved_model_dir (str): Source directory for the model to quantize.
            output_dir (str): Writable output directory to save the quantized model
            inc_config_path (str): Path to an INC config file (.yaml)

        Returns:
            None

        Raises:
            NotADirectoryError if the model is not a directory
            FileNotFoundError if a model.pt is not found in the model or if the inc_config_path file
            is not found.
            FileExistsError if the output_dir already has a model.pt file
        """
        # The saved model directory should exist and contain a model.pt file
        if not os.path.isdir(saved_model_dir):
            raise NotADirectoryError("The saved model directory ({}) does not exist.".format(saved_model_dir))
        if not os.path.isfile(os.path.join(saved_model_dir, "model.pt")):
            raise FileNotFoundError("The saved model directory ({}) should have a model.pt file".format(
                saved_model_dir))

        # Verify that the config file exists
        if not os.path.isfile(inc_config_path):
            raise FileNotFoundError("The config file was not found at: {}".format(inc_config_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Verify that the output directory doesn't already have a saved_model.pb file
            if os.path.exists(os.path.join(output_dir, "model.pt")):
                raise FileExistsError("A saved model already exists at:", os.path.join(output_dir, "model.pt"))

        from neural_compressor.experimental import Quantization

        quantizer = Quantization(inc_config_path)
        quantizer.model = self._model
        quantized_model = quantizer.fit()

        # If quantization was successful, save the model
        if quantized_model:
            quantized_model.save(output_dir)
            import subprocess
            # Change the model filename from best_model.pt to model.pt to match our convention
            p = subprocess.Popen(["mv", output_dir + "/best_model.pt", output_dir + "/model.pt"],
                                 stdout=subprocess.PIPE)
            stdout, stderr = p.communicate()

    def list_layers(self, verbose=False):
        """
        Lists all of the named modules (e.g. features, avgpool, classifier) and layers
        (ReLU, MaxPool2d, Dropout, Linear, etc) in a given PyTorch model

        Args:
            verbose (bool): True/False option set by default to be False, displays only high-level modules
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be summarized.')

        # Display a high-level list of the modules e.g. features, avgpool, classifier
        print("\nModel Layers\n============")
        for (name, module) in self._model.named_children():
            if not verbose or not list(module.named_children()):
                print('{}: {}/{} parameters are trainable'.format(
                    name, sum(p.numel() for p in module.parameters() if p.requires_grad),
                    sum(p.numel() for p in module.parameters())))
            else:
                print('{}:'.format(name))
                for (layer_name, layer) in module.named_children():
                    print('  {}: {}/{} parameters are trainable'.format(
                        layer_name, sum(p.numel() for p in layer.parameters() if p.requires_grad),
                        sum(p.numel() for p in layer.parameters())))

        trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print('\nTotal Trainable Parameters: {}/{}'.format(
            trainable_parameters,
            sum(p.numel() for p in self._model.parameters())))

        return trainable_parameters

    def freeze_layer(self, layer_name):
        """
        Freezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be frozen.')

        # Freeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = False

        return

    def unfreeze_layer(self, layer_name):
        """
        Unfreezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be unfrozen.')

        # Unfreeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = True

        return
