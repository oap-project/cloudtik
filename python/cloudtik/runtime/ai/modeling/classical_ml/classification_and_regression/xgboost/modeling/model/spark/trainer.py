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

from sparkxgb import XGBoostClassifier

xgb4j_spark_jar = '/workspace/third_party/xgboost4j-spark_2.12-1.5.2.jar'
xgb4j_jar = '/workspace/third_party/xgboost4j_2.12-1.5.2.jar'
os.environ['PYSPARK_SUBMIT_ARGS'] = f'--jars {xgb4j_spark_jar},{xgb4j_jar} pyspark-shell'


class XGBoostSparkClassifier:
    def __init__(self, xgb_params, label_col):
        self.xgb_params = xgb_params
        self.label_col = label_col 

    def fit(self, data):
        xgb_classifier = XGBoostClassifier(**self.xgb_params).setLabelCol(self.label_col)
        xgb_clf_model = xgb_classifier.fit(data)
        return xgb_clf_model 
