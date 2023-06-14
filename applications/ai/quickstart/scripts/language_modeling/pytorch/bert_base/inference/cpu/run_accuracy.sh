#!/bin/bash

#
# Copyright (c) 2021 Intel Corporation
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

ARGS=""

precision="fp32"
if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
else
    echo "### running fp32 mode"
fi

mode="default"
if [[ "$USE_IPEX" == "true" ]]; then
  ARGS="$ARGS --use_ipex --jit_mode"
  export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
  mode="jit"
fi

echo "Running using ${ARGS} args ..."

CORES=`lscpu | grep Core | awk '{print $4}'`
BATCH_SIZE=${BATCH_SIZE:-`expr 4 \* $CORES`}
FINETUNED_MODEL=${FINETUNED_MODEL:-"csarron/bert-base-uncased-squad-v1"}
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/question-answering/run_qa.py"}

rm -rf ${OUTPUT_DIR}/accuracy_log*
cloudtik-ai-run \
  --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --dataset_name squad \
  --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp \
  --per_device_eval_batch_size $BATCH_SIZE

match=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_exact_match" |sed -e 's/.*= //;s/[^0-9.]//g')
f1=$(cat ${OUTPUT_DIR}/accuracy_log* | grep "eval_f1" |sed -e 's/.*= //;s/[^0-9.]//g')
echo ""BERT-base";"exact_match";${precision};${BATCH_SIZE};${match}" | tee -a ${OUTPUT_DIR}/summary.log
echo ""BERT-base";"f1";${precision};${BATCH_SIZE};${f1}" | tee -a ${OUTPUT_DIR}/summary.log


