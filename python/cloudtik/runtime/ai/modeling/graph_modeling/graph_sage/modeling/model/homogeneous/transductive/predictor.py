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
    homogeneous.transductive.model import TransductiveGraphSAGEModel


class Predictor:
    def __init__(
            self, vocab_size, num_hidden, num_layers, model_file,
            mode, batch_size=10240):
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device("cpu" if mode == "cpu" else "cuda")

        self.model = TransductiveGraphSAGEModel(
            vocab_size, num_hidden, num_layers).to(self.device)
        self.model.eval()
        self.load_model(model_file)

    def predict(self, g):
        model = self.model

        g = g.to("cpu" if self.mode == "cpu" else "cuda")

        print("Inference to generate node representations...")

        # Since it is transductive, the entire embedding includes all nodes
        x = model.get_inference_inputs(g)
        node_emb = model.inference(
            g, x, self.device, self.batch_size)
        print("Node embeddings shape: ", node_emb.shape)
        return node_emb

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))


def predict(dataset_dir, num_hidden, num_layers, model_file,
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

    vocab_size = g.num_nodes()
    predictor = Predictor(
        vocab_size, num_hidden, num_layers,
        model_file=model_file, mode=mode, batch_size=batch_size)

    predictions = predictor.predict(g)

    if predict_output:
        # save the prediction output only if needed
        torch.save(predictions, predict_output)

    return predictions
