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

#

import os
import subprocess

BRCA_URLS = [
    {
        "file_name": "Medical reports for cases .zip",
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/109379611/Medical%20reports%20for%20cases%20.zip?api=v2"
    },
    {
        "file_name": "Radiology manual annotations.xlsx",
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/109379611/Radiology%20manual%20annotations.xlsx?api=v2"
    },
    {
        "file_name": "Radiology_hand_drawn_segmentations_v2.csv",
        "url": "https://wiki.cancerimagingarchive.net/download/attachments/109379611/Radiology_hand_drawn_segmentations_v2.csv?api=v2"
    }
]


def download(dataset_directory):
    if not dataset_directory:
        raise ValueError("Must specify the dataset directory to download to.")

    if not os.path.isdir(dataset_directory):
        os.makedirs(dataset_directory)

    # Get the URL for the dataset
    for url in BRCA_URLS:
        # Check if the key 'file_name' exists in the JSON file
        if 'file_name' in url:
            filename = url['file_name']
        else:
            filename = url['url'].split('/')[-1]

        destination_file_path = os.path.join(dataset_directory, filename)
        if os.path.exists(destination_file_path):
            print("\nFile already exists in {}.".format(destination_file_path))
            print("Please delete it and try again!.\n")
        else:
            # Download the file if it does not exist in the desired dataset directory
            subprocess.run(["wget", url['url'], "-O", destination_file_path])
            print("\n{} downloaded successfully in {}\n".format(filename, dataset_directory))
