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
from datetime import datetime
import simplejson as json
import glob

import xgboost as xgb
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType, RayShardingMode

from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import optuna

from ray import tune
from ray.tune.search.optuna import OptunaSearch

from .utils import partition_data

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
            on_ray=False, ray_params=None, hpo_spec=None):
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
        self.on_ray = on_ray
        self.ray_params = ray_params

        self.tmp_data_path = os.path.join(tmp_path, 'data')
        self.log_path = os.path.join(tmp_path, 'logs')

    def train(self):
        print("read and prepare data for training...")
        train, valid, test = self.split_data(self.df, self.data_split)

        if self.on_ray and not self.in_memory:
            train, valid, test = self.prepare_data(train, valid, test, 'csv')

        print("start xgboost model training...")
        if self.on_ray:
            xgb_model = XGBoostRay(
                train, valid, test, self.target_col,
                self.model_params, self.training_params, self.ray_params)
            self.model = xgb_model.fit()
        else:
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
        if self.on_ray:
            train_path, valid_path, test_path = self.prepare_data(train, valid, test, 'csv')

            print("start xgboost HPO...")
            xgb_model = XGBoostRay(
                train_path, valid_path, test_path, self.target_col,
                self.model_params, self.training_params, self.ray_params)
            xgb_model.tune(self.log_path)
            xgb_model.print_best_configs(self.test_metric)
            test_result = xgb_model.analysis.best_result[f"test-{self.test_metric}"]
            print(f"{self.test_metric} of the best configs on test set is {test_result}")
        else:
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

    def prepare_data(self, train_df, valid_df, test_df, data_format):
        partition_data(train_df, data_format, f"{self.tmp_data_path}/train", 100)
        partition_data(valid_df, data_format, f"{self.tmp_data_path}/valid", 100)
        partition_data(test_df, data_format, f"{self.tmp_data_path}/test", 100)

        train_path = list(sorted(glob.glob(f'{self.tmp_data_path}/train/*.{data_format}')))
        valid_path = list(sorted(glob.glob(f'{self.tmp_data_path}/valid/*.{data_format}')))
        test_path = list(sorted(glob.glob(f'{self.tmp_data_path}/test/*.{data_format}')))

        return train_path, valid_path, test_path

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


class XGBoost:
    def __init__(self,
                 train_df, valid_df, test_df, target_col,
                 model_params, training_params):
        self.dtrain = xgb.DMatrix(data=train_df.drop(target_col, axis=1), label=train_df[target_col])
        self.dvalid = xgb.DMatrix(data=valid_df.drop(target_col, axis=1), label=valid_df[target_col]) 
        self.dtest = xgb.DMatrix(data=test_df.drop(target_col, axis=1), label=test_df[target_col]) 
        self.watch_list = [(self.dtrain,'train'), (self.dvalid, 'eval'), (self.dtest, 'test')]
        self.model_params = model_params
        self.training_params = training_params
        try:
            self.epoch_log_interval = training_params['verbose_eval']
        except:
            self.epoch_log_interval = 25
        self.evals_result = []
    
    def fit(self):
        model = xgb.train(self.model_params, **self.training_params,
                          dtrain=self.dtrain, evals=self.watch_list)
        return model

    def _train_model(self, trial):
        params = {}
        params.update(self.fixed_model_params)
        for name, value in self.search_model_params.items():
            if name in ['eta', 'max_depth', 'subsample', 'colsample_bytree', 'lambda', 'alpha', 'min_child_weight', 'gamma']:
                if value['type'] == 'discrete':
                   params[name] = trial.suggest_categorical(name,value['boundary'])
                elif value['type'] == 'int':
                    params[name] = trial.suggest_int(name, value['boundary'][0], value['boundary'][1])
                elif value['type'] == 'float':
                    params[name] = trial.suggest_float(name, value['boundary'][0], value['boundary'][1])
                else:
                    raise ValueError('Please specify the correct type')
            else:
                raise NotImplementedError(
                    f"{name} is currently not supported.")

        evals_result = {}
        model = xgb.train(params,
                          dtrain=self.dtrain,
                          evals=self.watch_list,
                          evals_result=evals_result,
                          num_boost_round=self.num_boost_round,
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=self.num_boost_round - 1
                          )
        
        self.evals_result.append(evals_result)
        accuracy = evals_result["eval"][self.eval_metric][-1]
        return accuracy

    def tune(self, log_dir):
        num_trials = self.training_params['num_trials']
        search_mode = self.training_params['search_mode']
        if search_mode == 'max':
            search_mode = 'maximize'
        elif search_mode == 'min':
            search_mode = 'minimize'
        else:
            raise ValueError('only min or max is accepted for search_mode')
        
        try:
            self.num_boost_round = self.training_params['training_params']['num_boost_round']
        except:
            self.num_boost_round = 10

        try:
            self.early_stopping_rounds = self.training_params['training_params']['early_stopping_rounds']
        except:
            self.early_stopping_rounds = None

        self.fixed_model_params = self.model_params['fixed']
        self.search_model_params = self.model_params['search_space']
        self.eval_metric = self.fixed_model_params['eval_metric']

        study = optuna.create_study(direction=search_mode)
        study.optimize(self._train_model, n_trials=num_trials, show_progress_bar = True)
        self.best_trial = study.best_trial

    def save_best_configs(self, save_path):
        now = str(datetime.now().strftime("%Y-%m-%d+%H%M%S"))
        values = {'best accuracy': self.best_trial.value, 'best params': self.best_trial.params}
        with open(f'{save_path}/best_model_configs_{now}.json', 'w') as fp:
            json.dump(values, fp)

    def print_best_configs(self):
        print("  Value: {}".format(self.best_trial.value))
        print("  Params: ")
        for key, value in self.best_trial.params.items():
            print("    {}: {}".format(key, value))


class XGBoostRay:
    def __init__(self,
                 train, valid, test, target_col,
                 model_params, training_params, ray_params):
        self.dtrain = RayDMatrix(data=train, label=target_col)
        self.dvalid = RayDMatrix(data=valid, label=target_col)
        self.dtest = RayDMatrix(data=test, label=target_col)
        self.watch_list = [(self.dtrain, 'train'), (self.dvalid, 'eval'), (self.dtest, 'test')]
        self.model_params = model_params
        self.training_params = training_params
        try:
            self.epoch_log_interval = training_params['verbose_eval']
        except:
            self.epoch_log_interval = 25
        self.ray_params = RayParams(**ray_params) 
        self.evals_result = {}

    def fit(self):
        model = train(
            self.model_params, **self.training_params,
            dtrain=self.dtrain, evals=self.watch_list, ray_params=self.ray_params)
        return model

    def _train_xgb(self, config, ray_params):
        bst = train(
                params=config,
                dtrain=self.dtrain,
                evals=self.watch_list,
                evals_result=self.evals_result,
                verbose_eval=False,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                ray_params=ray_params)

    def tune(self, log_dir):
        try:
            self.num_boost_round = self.training_params['training_params']['num_boost_round']
        except:
            self.num_boost_round = 10

        try:
            self.early_stopping_rounds = self.training_params['training_params']['early_stopping_rounds']
        except:
            self.early_stopping_rounds = None 

        fixed_model_params = self.model_params['fixed']
        search_model_params = self.decode_search_params(self.model_params['search_space'])
        params = {**fixed_model_params, **search_model_params}

        metric = fixed_model_params['eval_metric']
        mode = self.training_params['search_mode']
        num_samples = self.training_params['num_trials']

        analysis = tune.run(
                tune.with_parameters(self._train_xgb, ray_params=self.ray_params),
                config=params,
                search_alg=OptunaSearch(metric=metric, mode=mode),
                num_samples=num_samples,
                metric=f"eval-{metric}", 
                mode=mode,
                local_dir = log_dir,
                max_failures=8,
                resources_per_trial=self.ray_params.get_tune_resources()
            )
        
        self.analysis = analysis 
    
    def print_best_configs(self, test_metric):
        accuracy = self.analysis.best_result[f"eval-{test_metric}"]
        print("  Value: {}".format(accuracy))
        print("  Params: ")
        for key, value in self.analysis.best_config.items():
            print("    {}: {}".format(key, value))

    def save_best_configs(self, save_path):
        now = str(datetime.now().strftime("%Y-%m-%d+%H%M%S"))
        values = {'best accuracy': self.analysis.best_result[f"eval-{self.test_metric}"], 
                  'best params': self.analysis.best_config}
        with open(f'{save_path}/best_model_configs_{now}.json', 'w') as fp:
            json.dump(values, fp)

    def decode_search_params(self, search_params):
        defined_search_params = {}
        for name, value in search_params.items():
            if name in ['eta', 'max_depth', 'subsample', 'colsample_bytree', 'lambda', 'alpha', 'min_child_weight', 'gamma']:
                if value['type'] == 'discrete':
                    defined_search_params[name] = tune.choice(value['boundary'])
                elif value['type'] == 'int':
                    defined_search_params[name] = tune.randint(value['boundary'][0], value['boundary'][1])
                elif value['type'] == 'float':
                    defined_search_params[name] = tune.uniform(value['boundary'][0], value['boundary'][1])
                else:
                    raise ValueError('Please specify the correct type')
            else:
                raise NotImplementedError(
                    f"{name} is currently not supported.")
        return defined_search_params 
