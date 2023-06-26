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

Author: Chen Haifeng
"""

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.distributed.model import DistGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.inductive.model import InductiveGraphSAGEModel


class DistInductiveGraphSAGEModel(DistGraphSAGEModel, InductiveGraphSAGEModel):
    def __init__(self, in_feats, hidden_size, num_layers, node_feature):
        InductiveGraphSAGEModel.__init__(
            self, in_feats, hidden_size, num_layers, node_feature)
