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

import torch
import dgl

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.predictor import Predictor
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.inductive.model import InductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.transductive.model import TransductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model. \
    heterogeneous.utils import get_in_feats_of_feature, get_node_types


def predict(dataset_dir, model_file,
            num_hidden=64, num_layers=2,
            relations=None,
            inductive=False,
            node_feature=None,
            predict_output=None,
            mode="cpu",
            batch_size=10240):
    if not torch.cuda.is_available():
        mode = "cpu"
    print(f"Predicting in {mode} mode.")

    dataset = dgl.data.CSVDataset(dataset_dir, force_reload=False)
    graph = dataset[0]  # only one graph

    # create model

    # Heterogeneous
    if relations:
        relations = [relation for relation in relations.split(",")]
    else:
        relations = graph.etypes

    if inductive:
        print("Predicting with an inductive model on heterogeneous graph")
        in_feats = get_in_feats_of_feature(graph, node_feature)
        model = InductiveGraphSAGEModel(
            in_feats, num_hidden, num_layers,
            relations=relations, node_feature=node_feature)
    else:
        print("Predicting with a transductive model on heterogeneous graph")
        # vocab_size is a dict of node type as key
        node_types = get_node_types(graph, relations)
        vocab_size = {k: graph.num_nodes(k) for k in node_types}
        model = TransductiveGraphSAGEModel(
            vocab_size, num_hidden, num_layers,
            relations=relations)

    predictor = Predictor(
        model, model_file=model_file,
        mode=mode, batch_size=batch_size)

    predictions = predictor.predict(graph)

    if predict_output:
        # save the prediction output only if needed
        torch.save(predictions, predict_output)

    return predictions
