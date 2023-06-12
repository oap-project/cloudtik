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

import optuna
import simplejson as json
import xgboost as xgb


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
