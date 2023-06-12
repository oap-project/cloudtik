"""
Copyright [2022-23] [Intel Corporation]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import glob

from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.ray.model import \
    XGBoostRay
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.utils import \
    partition_data
from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.trainer import \
    Trainer as XGBoostTrainer


class Trainer(XGBoostTrainer):
    def __init__(
            self, data_spec, df, model_spec,
            in_memory, tmp_path,
            hpo_spec=None,
            ray_params=None):
        super().__init__(
            data_spec, df, model_spec,
            in_memory, tmp_path, hpo_spec)

        self.ray_params = ray_params

    def train(self):
        print("read and prepare data for training...")
        train, valid, test = self.split_data(self.df, self.data_split)

        if not self.in_memory:
            train, valid, test = self.prepare_data(train, valid, test, 'csv')

        print("start xgboost model training...")
        xgb_model = XGBoostRay(
            train, valid, test, self.target_col,
            self.model_params, self.training_params, self.ray_params)
        self.model = xgb_model.fit()

    def train_hpo(self):
        print("read and prepare data for training...")
        train, valid, test = self.split_data(self.df, self.data_split)

        print("start xgboost HPO...")
        train_path, valid_path, test_path = self.prepare_data(train, valid, test, 'csv')

        print("start xgboost HPO...")
        xgb_model = XGBoostRay(
            train_path, valid_path, test_path, self.target_col,
            self.model_params, self.training_params, self.ray_params)
        xgb_model.tune(self.log_path)
        xgb_model.print_best_configs(self.test_metric)
        test_result = xgb_model.analysis.best_result[f"test-{self.test_metric}"]
        print(f"{self.test_metric} of the best configs on test set is {test_result}")

    def prepare_data(self, train_df, valid_df, test_df, data_format):
        partition_data(train_df, data_format, f"{self.tmp_data_path}/train", 100)
        partition_data(valid_df, data_format, f"{self.tmp_data_path}/valid", 100)
        partition_data(test_df, data_format, f"{self.tmp_data_path}/test", 100)

        train_path = list(sorted(glob.glob(f'{self.tmp_data_path}/train/*.{data_format}')))
        valid_path = list(sorted(glob.glob(f'{self.tmp_data_path}/valid/*.{data_format}')))
        test_path = list(sorted(glob.glob(f'{self.tmp_data_path}/test/*.{data_format}')))

        return train_path, valid_path, test_path


def train(
        data_spec, df, model_spec,
        in_memory, tmp_path, model_file,
        hpo_spec=None,
        ray_params=None):
    # start ray client
    import ray
    ray.init(
        'auto',
        runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}},
        log_to_driver=False)

    trainer = Trainer(data_spec, df,
                      model_spec, in_memory,
                      tmp_path=tmp_path, hpo_spec=hpo_spec,
                      ray_params=ray_params)

    if hpo_spec is not None:
        trainer.train_hpo()
    else:
        trainer.train()
        if model_file:
            trainer.save_model(model_file)

    # shutdown ray client
    ray.shutdown()
