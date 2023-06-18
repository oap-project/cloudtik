#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

from disease_prediction.dlsa.data.process import process as process_dlsa
from disease_prediction.vision.data.process import process as process_vision


def get_output_annotations_file(output_dir):
    return os.path.join(
        output_dir, "annotation", "annotation.csv")


def process(data_path, image_path, output_dir):
    output_annotations_file = get_output_annotations_file(
        output_dir)
    process_dlsa(
        data_path=data_path,
        output_annotations_file=output_annotations_file)
    process_vision(
        data_path=data_path,
        image_path=image_path,
        output_dir=output_dir)

