#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/train_net.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at the: \${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/train_net.py"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/coco" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/coco does not exist"
  exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "The OUTPUT_DIR '${OUTPUT_DIR}' does not exist"
  exit 1
fi

if [[ "$1" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""

if [[ "$1" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
elif [[ "$1" == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
elif [[ "$1" == "fp32" || "$1" == "avx-fp32" ]]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, and bf32."
    exit 1
fi

if [[ "$USE_IPEX" == "true" ]]; then
  export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
fi

BACKEND=${BACKEND:-'ccl'}

if [[ "$BACKEND" == "ccl" ]]; then
  oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
  source $oneccl_bindings_for_pytorch_path/env/setvars.sh
fi

CORES=${CORES:-`lscpu | grep Core | awk '{print $4}'`}
SOCKETS=${SOCKETS:-`lscpu | grep Socket | awk '{print $2}'`}
TOTAL_CORES=`expr $CORES \* $SOCKETS`

CORES_PER_INSTANCE=$CORES

HOSTS=${HOSTS:-'127.0.0.1'}
NNODES=$(echo $HOSTS | tr ',' '\n' | wc -l)
NUM_RANKS=$(( NNODES * SOCKETS ))

export TRAIN=1

PRECISION=$1

BATCH_SIZE=${BATCH_SIZE-112}

rm -rf ${OUTPUT_DIR}/distributed_throughput_log_${PRECISION}*

cloudtik-run \
    --distributed \
    --nnodes ${NNODES} \
    --hosts ${HOSTS} \
    --nproc_per_node $SOCKETS \
    --log-dir=${OUTPUT_DIR} \
    --log-file-prefix="./distributed_throughput_log_${PRECISION}" \
    ${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/train_net.py \
    $ARGS \
    --iter-warmup 10 \
    -i 20 \
    --config-file "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_tra.yaml" \
    --skip-test \
    --backend $BACKEND \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.MAX_ITER 720000 \
    SOLVER.STEPS '"(60000, 80000)"' \
    SOLVER.BASE_LR 0.0025 \
    MODEL.DEVICE cpu \
    2>&1 | tee ${OUTPUT_DIR}/distributed_throughput_log_${PRECISION}.txt

# For the summary of results
wait
throughput=$(grep 'Training throughput:' ${OUTPUT_DIR}/distributed_throughput_log_${PRECISION}* |sed -e 's/.*Training throughput//;s/[^0-9.]//g' |awk '
BEGIN {
        sum = 0;
        i = 0;
      }
      {
        sum = sum + $1;
        i++;
      }
END   {
        sum = sum / i;
        printf("%.3f", sum);
}')
echo ""maskrcnn";"training distributed throughput";${PRECISION};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log

