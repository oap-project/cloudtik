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

import numpy as np  


class DataTransformer:
    def __init__(self, df, steps, data_api):
        self.df = df 
        self.steps = steps 
        self.tmp = {}
        self.data_api = data_api

    def transform(self):
        for step in self.steps:
            op = list(step.keys())[0]
            if op == 'normalize_feature_names':
                self.normalize_feature_names(list(step.values())[0])
            elif op == 'rename_feature_names':
                raise NotImplementedError
            elif op == 'drop_features':
                raise NotImplementedError
            elif op == 'outlier_treatment':
                raise NotImplementedError
            elif op == 'categorify':
                self.categorify(list(step.values())[0])
            elif op == 'strip_chars':
                self.strip_chars(list(step.values())[0])
            elif op == 'combine_cols':
                self.combine_cols(list(step.values())[0])
            elif op == 'change_datatype':
                self.change_datatype(list(step.values())[0])
            elif op == 'time_to_seconds':
                self.time_to_seconds(list(step.values())[0])
            elif op == 'min_max_normalization':
                self.min_max_normalization(list(step.values())[0])
            elif op == 'one_hot_encoding':
                self.one_hot_encoding(list(step.values())[0])
            elif op == 'string_to_list':
                self.string_to_list(list(step.values())[0])
            elif op == 'multi_hot_encoding':
                self.multi_hot_encoding(list(step.values())[0])
            elif op == 'add_constant_feature':
                self.add_constant_feature(list(step.values())[0])
            elif op == 'define_variable':
                self.define_variable(list(step.values())[0])
            elif op == 'modify_on_conditions':
                self.modify_on_conditions(list(step.values())[0])
                
        return self.df 

    def normalize_feature_names(self, steps):
        for step in steps:
            op = list(step.keys())[0]
            if op == 'replace_chars':
                self.replace_chars(list(step.values())[0])
            elif op == 'lowercase':
                if list(step.values())[0]:
                    self.to_lowercase()

    def replace_chars(self, replacements):
        for key, value in replacements.items():
            self.df.columns = self.df.columns.str.replace(key, value)

    def to_lowercase(self):
        self.df.columns = self.df.columns.str.lower()

    def categorify(self, features):
        for target_feature, new_feature in features.items():
            self.df[new_feature] = self.df[target_feature].astype('category').cat.codes

    def strip_chars(self, features):
        for old_feature, mapping in features.items():
            for new_feature, char in mapping.items():
                self.df[new_feature] = self.df[old_feature].str.strip(char)
    
    def combine_cols(self, features):
        for new_feature, content in features.items():
            for operation, target_feature_list in content.items(): 
                if len(target_feature_list) < 2:
                    raise ValueError('there is less than 2 items in the list, cannot concatenate')
                else:
                    if operation == 'concatenate_strings':
                        tmp_feature = self.df[target_feature_list[0]].astype('str')
                        for feature in target_feature_list[1:]:
                            tmp_feature = tmp_feature + self.df[feature].astype('str')
                        self.df[new_feature] = tmp_feature
    
    def change_datatype(self, col_dtypes):
        for col, dtype in col_dtypes.items():
            if isinstance(dtype, list):
                for type in dtype:
                    self.df[col] = self.df[col].astype(type)
            else:
                self.df[col] = self.df[col].astype(dtype)

    def time_to_seconds(self, features):
        for old_feature, new_feature in features.items():
            self.df[old_feature] = self.df[old_feature].astype('datetime64[s]')
            self.df[new_feature] = self.df[old_feature].dt.hour*60 + self.df[old_feature].dt.minute 
    
    def min_max_normalization(self, features):
        for old_feature, new_feature in features.items():
            self.df[new_feature] = (self.df[old_feature] - self.df[old_feature].min())/(
                    self.df[old_feature].max() - self.df[old_feature].min())
         
    def one_hot_encoding(self, features):
        pd = self.data_api.pandas()
        for feature, is_drop in features.items():
            self.df = pd.concat([self.df, pd.get_dummies(self.df[[feature]])], axis=1)
            if is_drop:
                self.df.drop(columns=[feature], axis=1, inplace=True)

    def string_to_list(self, features):
        for old_feature, mapping in features.items():
            for new_feature, sep in mapping.items():
                self.df[new_feature] = self.df[old_feature].map(lambda x: str(x).split(sep))

    def multi_hot_encoding(self, features):
        pd = self.data_api.pandas()
        # TODO: It's critical that the API implementation should compatible
        #   why the code needs to handle specially for explode, groupby, and column names?
        if self.data_api.native:
            for feature, is_drop in features.items():
                exploded = self.df[feature].explode()
                raw_one_hot = pd.get_dummies(exploded, columns=[feature])
                tmp_df = raw_one_hot.groupby(raw_one_hot.index).sum()
                self.df = pd.concat([self.df, tmp_df], axis=1)
                col_names = self.df.columns
                if '' in col_names or 'nan' in col_names:
                    self.df.drop(columns=['', 'nan'], axis=1, inplace=True)
                if is_drop:
                    self.df.drop(columns=[feature], axis=1, inplace=True)
        else:
            for feature, is_drop in features.items():
                exploded = self.df[feature].explode().to_frame()
                raw_one_hot = pd.get_dummies(exploded, columns=[feature])
                tmp_df = raw_one_hot.groupby(level=0).sum()
                self.df = pd.concat([self.df, tmp_df], axis=1) 
                to_be_replaced = feature.split('?')[0]
                self.df.columns = self.df.columns.str.replace(to_be_replaced+'\?'+'_', '')
                col_names = self.df.columns 
                if '' in col_names or 'nan' in col_names: 
                    self.df.drop(columns=['', 'nan'], axis=1, inplace=True)
                if is_drop: 
                    self.df.drop(columns=[feature], axis=1, inplace=True)
    
    def add_constant_feature(self, features):
        pd = self.data_api.pandas()
        # TODO: It's critical that the API implementation should compatible
        #   is this a bug to not respect the const_value for other cases?
        if self.data_api.native:
            for target_feature, const_value in features.items():
                self.df[target_feature] = const_value
        else:
            for target_feature, const_value in features.items():
                self.df[target_feature] = pd.Series(np.zeros(self.df.shape[0]), dtype=np.int8)

    def define_variable(self, definitions):
        df = self.df
        for var_name, expression in definitions.items():
            self.tmp[var_name] = eval(expression)

    def modify_on_conditions(self, map):
        tmp = self.tmp 
        for col, conditions in map.items():
            for condition, value in conditions.items():
                df = self.df
                df.loc[eval(condition), col] = value 
                self.df = df 
