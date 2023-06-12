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

from datetime import datetime

import simplejson as json
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from xgboost_ray import RayDMatrix, RayParams, train


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
