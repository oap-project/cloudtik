#!/bin/bash

# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Examples:
#   ./prepare_dataset.sh

set -e
set -x

ls -ltrash

# specify download_dir to be the location directory of your downloaded dataset
download_dir=${download_dir:-"$HOME/data/dlrm/criteo"}
# ./verify-criteo-downloaded.sh ${download_dir}

output_path=${output_path:-"$HOME/data/dlrm/output"}

if [ -f ${output_path}/train/_SUCCESS ] \
    && [ -f ${output_path}/validation/_SUCCESS ] \
    && [ -f ${output_path}/test/_SUCCESS ]; then
    echo "Spark preprocessing already carried out"
else
    echo "Performing Spark preprocessing"
    bash run-spark-cpu.sh ${download_dir} ${output_path}
fi

# download processed data and convert to custom binary format
hadoop fs -get $HOME/data/dlrm/output $HOME/data/dlrm/

conversion_intermediate_dir=${conversion_intermediate_dir:-"$HOME/data/dlrm/intermediate_binary"}
final_output_dir=${final_output_dir:-"$HOME/data/dlrm/binary_dataset"}

if [ -d ${final_output_dir}/train ] \
   && [ -d ${final_output_dir}/validation ] \
   && [ -d ${final_output_dir}/test ] \
   && [ -f ${final_output_dir}/feature_spec.yaml ]; then

    echo "Final conversion already done"
else
    echo "Performing final conversion to a custom data format"
    export TOTAL_CORES=$(cloudtik resources --cpu)
    python parquet_to_binary.py --parallel_jobs ${TOTAL_CORES} --src_dir ${output_path} \
                                --intermediate_dir  ${conversion_intermediate_dir} \
                                --dst_dir ${final_output_dir}

    cp "${output_path}/model_size.json" "${final_output_dir}/model_size.json"

    git clone https://github.com/NVIDIA/DeepLearningExamples  $HOME/DeepLearningExamples
    python split_dataset.py --dataset "${final_output_dir}" --output "${final_output_dir}/split"
    rm ${final_output_dir}/train_data.bin
    rm ${final_output_dir}/validation_data.bin
    rm ${final_output_dir}/test_data.bin
    rm ${final_output_dir}/model_size.json

    mv ${final_output_dir}/split/* ${final_output_dir}
    rm -rf ${final_output_dir}/split
fi

echo "Done preprocessing the Criteo Terabyte dataset"



