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

from category_encoders import TargetEncoder
from sklearn import preprocessing
from scipy.special import expit


class TargetEncoderModin:
    def __init__(self, input_col=None, target_col=None, min_samples_leaf=20, smoothing=10):
        self.input_col = input_col
        self.target_col = target_col
        self.min_samples_leaf = min_samples_leaf 
        self.smoothing = smoothing       
        self.mapping = None
        self._mean = None
        
    def fit(self, X):
        mapping = {}
        
        y = X[self.target_col]
        scalar = self._mean = y.mean()       
        
        if X[self.input_col].dtype.name == 'category':
            X[self.input_col] = X[self.input_col].cat.codes
        
        stats = y.to_frame().groupby(X[self.input_col]).agg({self.target_col:['count', 'mean']})
        stats.columns = stats.columns.droplevel(0)
        
        smoove = self._weighting(stats['count'])
        
        smoothing = scalar * (1-smoove) + stats['mean'] * smoove
        smoothing.loc[-12] = scalar
        mapping[self.input_col] = smoothing 

        return mapping 
    
    def transform(self, X):
        if X[self.input_col].dtype.name == 'category':
            result = X[self.input_col].cat.codes.map(self.mapping[self.input_col])
        else:
            result = X[self.input_col].map(self.mapping[self.input_col])

        if result.isnull().sum() > 0:
            result.fillna(self.mapping[self.input_col].loc[-12], inplace=True)
        return result

    def fit_transform(self, X):
        _X = X.copy()
        self.mapping = self.fit(_X)
        return self.transform(X)

    def _weighting(self, n):
        tmp = (n - self.min_samples_leaf) / self.smoothing
        res = tmp.apply(lambda sr: expit(sr))
        return res


class PostTransformer:
    def __init__(self, train_data, test_data, steps, dp_engine):
        self.train_data = train_data
        self.test_data = test_data  
        self.steps = steps 
        self.dp_engine = dp_engine

    def transform(self):
        for step in self.steps:
            match list(step.keys())[0]: 
                case 'target_encoding': 
                    self.target_encoding(list(step.values())[0])
                case 'label_encoding':
                    self.label_encoding(list(step.values())[0])
        return self.train_data, self.test_data  

    def target_encoding(self, params):
        target_col = params['target_col']
        feature_cols = params['feature_cols']
        smoothing = params['smoothing']

        if self.dp_engine == 'modin':
            for col in feature_cols:
                tgt_encoder = TargetEncoderModin(input_col=col, target_col=target_col, smoothing=smoothing)
                self.train_data[col] = tgt_encoder.fit_transform(
                    self.train_data)
                self.test_data[col] = tgt_encoder.transform(
                    self.test_data)
        else:
            for col in feature_cols:
                tgt_encoder = TargetEncoder(smoothing=smoothing)
                self.train_data[col] = tgt_encoder.fit_transform(
                    self.train_data[col], self.train_data[target_col]).astype('float32')
                self.test_data[col] = tgt_encoder.transform(
                    self.test_data[col], self.test_data[target_col]).astype('float32')

    def label_encoding(self, params):
        feature_cols = params['feature_cols']
        for col in feature_cols: 
            label_encoder = preprocessing.LabelEncoder()
            self.train_data[col] = label_encoder.fit_transform(
                self.train_data[col]).astype('int64')
            self.test_data[col] = label_encoder.transform(
                self.test_data[col]).astype('int64')
