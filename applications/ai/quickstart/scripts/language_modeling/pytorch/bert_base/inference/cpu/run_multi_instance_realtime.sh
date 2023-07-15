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

export OMP_NUM_THREADS=4
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
BATCH_SIZE=${BATCH_SIZE:-1}
FINETUNED_MODEL=${FINETUNED_MODEL:-"csarron/bert-base-uncased-squad-v1"}
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/pytorch/question-answering/run_qa.py"}

rm -rf ${OUTPUT_DIR}/latency_log*
cloudtik-run \
  --latency_mode --memory-allocator=jemalloc --log-dir=${OUTPUT_DIR} --log-file-prefix="./latency_log_${precision}_${mode}" \
  ${EVAL_SCRIPT} $ARGS \
  --model_name_or_path   ${FINETUNED_MODEL} \
  --dataset_name squad \
  --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp \
  --max_eval_samples 100 \
  --per_device_eval_batch_size $BATCH_SIZE \

CORES_PER_PROC=4
TOTAL_CORES=`expr $CORES \* $SOCKETS`
PROCESSES=`expr $TOTAL_CORES / $CORES_PER_PROC`
PROCESSES_PER_SOCKET=`expr $PROCESSES / $SOCKETS`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/latency_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v PROCESSES_PER_SOCKET=$PROCESSES_PER_SOCKET '
BEGIN {
        sum = 0;
i = 0;
      }
      {
        sum = sum + $1;
i++;
      }
END   {
sum = sum / i * PROCESSES_PER_SOCKET;
        printf("%.2f", sum);
}')

echo $PROCESSES_PER_SOCKET
echo ""BERT-base";"latency";${precision};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
