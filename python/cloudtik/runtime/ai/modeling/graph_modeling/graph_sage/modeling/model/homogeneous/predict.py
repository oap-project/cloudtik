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

import torch
import dgl

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.predictor import Predictor
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.transductive.model import TransductiveGraphSAGEModel
from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    homogeneous.inductive.model import InductiveGraphSAGEModel


def predict(dataset_dir, model_file,
            num_hidden=64, num_layers=2,
            inductive=False,
            node_feature=None,
            predict_output=None,
            mode="cpu",
            batch_size=10240):
    if not torch.cuda.is_available():
        mode = "cpu"
    print(f"Training in {mode} mode.")

    dataset = dgl.data.CSVDataset(dataset_dir, force_reload=False)
    graph = dataset[0]  # only one graph

    # Assuming we are doing transductive training and prediction
    # to be consistent if there are new data
    # shall we join the new data into the graph for predicting?
    g = dgl.to_homogeneous(graph)

    # create model
    if inductive:
        model = InductiveGraphSAGEModel(
            node_feature, num_hidden, num_layers)
    else:
        vocab_size = g.num_nodes()
        model = TransductiveGraphSAGEModel(
            vocab_size, num_hidden, num_layers)

    predictor = Predictor(
        model, model_file=model_file,
        mode=mode, batch_size=batch_size)

    predictions = predictor.predict(g)

    if predict_output:
        # save the prediction output only if needed
        torch.save(predictions, predict_output)

    return predictions
