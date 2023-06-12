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

import os

import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.model \
    import XGBoost

"""
data_spec example:
  target_col: is_fraud?
  ignore_cols: ['user', 'card', 'merchant_name', 'split']
  data_split:
    train: df[df["year"]<2018]
    valid: df[df["year"]==2018]
    test: df[df["year"]>2018]

Either specify model_spec or specify hpo_spec.

model_spec examples:
  model_type: xgboost
  model_params: 
    learning_rate: 0.1
    eval_metric: 'aucpr'
    objective: 'binary:logistic'
    tree_method: 'hist'
    random_state: 42
    max_depth: 8
    subsample:  0.599094
    colsample_bytree: 0.692394
    lambda: 0.921488
    alpha: 0.329058
    min_child_weight: 7
  training_params:
    num_boost_round: 1000
    verbose_eval: 100
  test_metric: 'aucpr'

hpo_spec example:
  model_type: xgboost 
  model_params:
    fixed:
      objective: 'binary:logistic'
      tree_method: 'hist'
      eval_metric: 'aucpr'
      random_state: 42
    search_space:
      eta: 
        type: discrete #Sample from a given list    
        boundary: [0.01, 0.02, 0.03, 0.04, 0.05] 
      max_depth:
        type: int #Sample a integer uniformly between 1 (inclusive) and 9 (exclusive)
        boundary: [8, 9]
      subsample: 
        type: float 
        boundary: [0.5, 1.0] #Sample a float uniformly between 0.5 and 1.0
      colsample_bytree:
        type: float
        boundary: [0.2, 1]
      lambda:
        type: float
        boundary: [0.00000001, 1]
      alpha:
        type: float
        boundary: [0.00000001, 1]
      min_child_weight:
        type: int
        boundary: [7, 10]
  training_params: 
    num_boost_round: 1000
  test_metric: 'aucpr'
  search_mode: 'max'
  num_trials: 10    
"""


class Trainer:
    def __init__(
            self, data_spec, df, model_spec,
            in_memory, tmp_path,
            hpo_spec=None):
        self.target_col = data_spec['target_col']
        try:
            self.ignore_cols = data_spec['ignore_cols']
        except:
            self.ignore_cols = None
        self.data_split = data_spec['data_split']
        self.df = df.drop(columns=self.ignore_cols) if df is not None else None
        print(self.df.shape) 

        if model_spec is not None: 
            self.model_type = model_spec['model_type']
            self.model_params = model_spec['model_params']
            self.training_params = model_spec['training_params']
            self.test_metric = model_spec['test_metric']
            objective = model_spec['model_params']['objective']
            self.problem_type = 'classification' if 'binary' in objective or 'multi' in objective else 'regression'
        else:
            self.model_type = hpo_spec['model_type']
            fixed_model_params = hpo_spec['model_params']['fixed']
            search_params = hpo_spec['model_params']['search_space']
            self.model_params = {'fixed': fixed_model_params, 'search_space': search_params}
            self.test_metric = hpo_spec['test_metric']
            self.training_params = {'search_mode': hpo_spec['search_mode'], 'num_trials': hpo_spec['num_trials'], 
                                    'training_params': hpo_spec['training_params']}
        self.in_memory = in_memory

        self.tmp_data_path = os.path.join(tmp_path, 'data')
        self.log_path = os.path.join(tmp_path, 'logs')

    def train(self):
        print("read and prepare data for training...")
        train, valid, test = self.split_data(self.df, self.data_split)

        print("start xgboost model training...")
        xgb_model = XGBoost(
            train, valid, test, self.target_col,
            self.model_params, self.training_params)
        self.model = xgb_model.fit()

        print("start xgboost model testing...")
        test_result = self.test_model(test, self.target_col, self.test_metric)
        print(f"testing results: {self.test_metric} on test set is {test_result}")

    def train_hpo(self):
        print("read and prepare data for training...")
        train, valid, test = self.split_data(self.df, self.data_split)

        print("start xgboost HPO...")
        xgb_model = XGBoost(
            train, valid, test, self.target_col, self.model_params, self.training_params)
        xgb_model.tune(self.log_path)
        xgb_model.print_best_configs()
        xgb_model.save_best_configs(self.log_path)
        trial_id = xgb_model.best_trial._trial_id
        best_value_idx = xgb_model.evals_result[trial_id]["eval"][self.test_metric].index(xgb_model.best_trial.value)
        test_result = xgb_model.evals_result[trial_id]["test"][self.test_metric][best_value_idx]

    def split_data(self, df, data_split):
        train = data_split['train']
        valid = data_split['valid']
        test = data_split['test']

        train_df = eval(train)
        valid_df = eval(valid)
        test_df = eval(test)

        return train_df, valid_df, test_df

    def test_model(self, test_df, label, test_metric):
        dtest = xgb.DMatrix(data=test_df.drop(label, axis=1), label=test_df[label])
        probs = self.model.predict(dtest)

        if test_metric == 'aucpr':
            precision, recall, _ = precision_recall_curve(test_df[label], probs)
            test_result = auc(recall, precision)
        else:
            raise NotImplementedError('currently only aucrpr is supported as testing metric')
        
        return test_result 

    def save_model(self, model_file):
        self.model.save_model(model_file)
        print(f"{self.model_type} model is saved to {model_file}.")


def train(
        data_spec, df, model_spec,
        in_memory, tmp_path, model_file,
        hpo_spec=None):

    trainer = Trainer(
        data_spec, df,
        model_spec, in_memory,
        tmp_path=tmp_path, hpo_spec=hpo_spec)

    if hpo_spec is not None:
        trainer.train_hpo()
    else:
        trainer.train()
        if model_file:
            trainer.save_model(model_file)
