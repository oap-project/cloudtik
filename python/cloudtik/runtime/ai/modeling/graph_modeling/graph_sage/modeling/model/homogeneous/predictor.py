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


class Predictor:
    def __init__(
            self, model, model_file,
            mode, batch_size=10240):
        self.model = model
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device("cpu" if mode == "cpu" else "cuda")
        self.model.eval()
        self.load_model(model_file)

    def predict(self, g):
        model = self.model.to(self.device)

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
