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

from cloudtik.runtime.ai.modeling.graph_modeling.graph_sage.modeling.model.\
    heterogeneous.utils import tensor_dict_shape


class Predictor:
    def __init__(
            self, model, model_file, mode="cpu", batch_size=10240):
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device("cpu" if mode == "cpu" else "cuda")
        self.model = model.to(self.device)
        self.model.eval()
        self.load_model(model_file)

    def predict(self, g):
        model = self.model

        g = g.to("cpu" if self.mode == "cpu" else "cuda")

        print("Inference to generate node representations...")

        x = model.get_inference_inputs(g)
        node_emb = model.inference(
            g, x, self.device, self.batch_size)
        print("Node embeddings shape:", str(tensor_dict_shape(node_emb)))
        return node_emb

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))
