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


class DataSplitter:
    def __init__(self, df, param):
        self.df = df 
        self.param = param 

    def split(self):
        op = list(self.param.keys())[0]
        if op == 'custom_rules':
            self.custom_rules(list(self.param.values())[0])
        elif op == 'random_split':
            self.random_split(list(self.param.values())[0])
        return self.train_data, self.test_data  

    def custom_rules(self, rules):
        train_rules = rules['train']
        test_rules = rules['test']
        df = self.df
        self.train_data = df[eval(train_rules)]
        self.test_data = df[eval(test_rules)]

    def random_split(self, params):
        pass
