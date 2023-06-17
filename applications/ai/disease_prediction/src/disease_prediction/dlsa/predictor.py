# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#


from datasets import load_dataset, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer as TransformerTrainer
)

from disease_prediction.dlsa.utils import Benchmark, compute_metrics, save_test_metrics


class Predictor(object):
    def __init__(self, **kwargs):
        self.args = kwargs["args"]
        self.training_args = kwargs["training_args"]

        self.max_train, self.max_test = (
            self.args.max_train_samples,
            self.args.max_test_samples,
        )
        if self.args.smoke_test:
            self.max_train, self.max_test = 100, 100

        self.bench = Benchmark()
        self.track = self.bench.track

    def predict(self):
        with self.track("Total Run"):
            self._load_data(self.args.dataset, self.args.dataset_config)
            self._preprocess()
            self._load_model()
            self._predict()

    def _load_data(self, dataset, dataset_config=None):
        with self.track("Load Data"):
            self.remove_columns = []
            self.text_column = []
            if dataset == "local":
                class_names = dataset_config.label_list
                self.num_labels = len(class_names)
                customer_features = {v: k for k, v in dataset_config.features.items()}

                for key, value in customer_features.items():
                    if value == "class_label":
                        customer_features[key] = ClassLabel(names=class_names)
                    elif value == "data_column":
                        customer_features[key] = Value("string")
                        self.text_column = key
                        self.remove_columns.append(key)
                    else:
                        customer_features[key] = Value("string")
                        self.remove_columns.append(key)

                features = Features(customer_features)

                test_all = load_dataset(
                    "csv",
                    data_files=dataset_config.test,
                    delimiter=dataset_config.delimiter,
                    features=features,
                    split="train",
                )
                test_data = (
                    test_all.select(range(self.max_test)) if self.max_test else test_all
                )

                label2id = {class_names[i]: i for i in range(len(class_names))}
                self.test_data = test_data.align_labels_with_mapping(label2id, "label")
            else:
                data = load_dataset(dataset)
                test_split = "validation" if dataset == "sst2" else "test"
                if self.args.multi_instance:
                    start_index = (
                        self.args.instance_index - 1
                    ) * self.args.max_test_samples
                    end_index = self.args.instance_index * self.args.max_test_samples
                    self.test_data = data[test_split].select(
                        range(start_index, end_index)
                    )
                    print("start_index is ", start_index)
                    print("end_index is ", end_index)
                    print("test length is ", len(self.test_data))
                else:
                    self.test_data = (
                        data[test_split].select(range(self.max_test))
                        if self.max_test
                        else data[test_split]
                    )

                self.text_column = [
                    c
                    for c in self.test_data.column_names
                    if type(self.test_data[c][0]) != int
                ][0]

    def _preprocess(self):
        with self.track("Pre-process"):
            with self.track("----Init tokenizer"):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_name
                    if self.args.tokenizer_name
                    else self.args.model_name_or_path
                )

            max_seq_len = min(self.args.max_seq_len, self.tokenizer.model_max_length)

            with self.track("----Tokenize + Extract Features"):
                def preprocess(examples):
                    return self.tokenizer(
                        examples[self.text_column],
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_len,
                    )

                kwargs = dict(
                    function=preprocess,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=self.remove_columns,
                    load_from_cache_file=not self.args.overwrite_cache,
                )

                self.test_data = (
                    self.test_data.map(**kwargs)
                )

    def _load_model(self):
        with self.track('Load Model'):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_name_or_path)

            self.trainer = TransformerTrainer(
                model=self.model,  # the instantiated HF model to be trained
                args=self.training_args,  # training arguments, defined above
                compute_metrics=compute_metrics,  # evaluation metrics
                tokenizer=self.tokenizer
            )

    def _predict(self):
        with self.track('Inference'):
            preds, _, metrics = self.trainer.predict(self.test_data)
            test_metrics = save_test_metrics(metrics, len(self.test_data), self.training_args.output_dir)
        print(test_metrics)
