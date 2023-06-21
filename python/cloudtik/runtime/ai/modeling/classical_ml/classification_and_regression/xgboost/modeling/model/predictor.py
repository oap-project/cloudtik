"""
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

import numpy as np
import xgboost as xgb


class Predictor:
    def __init__(
            self, data_spec, model_file):
        self.target_col = None
        self.ignore_cols = None
        if data_spec is not None:
            if 'target_col' in data_spec:
                self.target_col = data_spec['target_col']
            if 'ignore_cols' in data_spec:
                self.ignore_cols = data_spec['ignore_cols']
        self.model = xgb.Booster()
        self.load_model(model_file)

    def predict(self, test_data):
        test_data = self._process(test_data)
        dtest = xgb.DMatrix(data=test_data)
        probs = self.model.predict(dtest)
        return probs

    def load_model(self, model_file):
        self.model.load_model(model_file)

    def _process(self, test_data):
        # drop ignore columns and target cols if exists
        valid_columns = set(test_data.columns)
        if self.ignore_cols or self.target_col:
            columns_to_drop = []
            if self.target_col and self.target_col in valid_columns:
                columns_to_drop.append(self.target_col)
            if self.ignore_cols:
                for ignore_col in self.ignore_cols:
                    if ignore_col in valid_columns:
                        columns_to_drop.append(self.target_col)
            if columns_to_drop:
                test_data = test_data.drop(columns=columns_to_drop)
        return test_data


def predict(data, model_file,
            data_spec=None, predict_output=None):
    predictor = Predictor(data_spec, model_file)
    predictions = predictor.predict(data)

    if predict_output:
        # save the prediction output
        np.savetxt(predict_output, predictions)

    return predictions
