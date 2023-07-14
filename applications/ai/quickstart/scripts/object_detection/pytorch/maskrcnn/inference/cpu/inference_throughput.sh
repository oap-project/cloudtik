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

if [ ! -e "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py" ]; then
  echo "Could not find the script test_net.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the test_net.py exist at: \${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth does not exist"
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
    if [[ "$2" == "jit" ]]; then
        ARGS="$ARGS --jit"
        echo "### running jit mode"
    elif [[ "$2" == "imperative" ]]; then
        echo "### running imperative mode"
    else
        echo "The specified mode '$2' is unsupported."
        echo "Supported mode are: imperative and jit."
        exit 1
    fi
    export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
fi

export TRAIN=0

PRECISION=$1

source "${MODEL_DIR}/scripts/utils.sh"
_get_platform_type

if [[ ${PLATFORM} == "windows" ]]; then
  CORES="${NUMBER_OF_PROCESSORS}"
else
  CORES=`lscpu | grep Core | awk '{print $4}'`
fi
BATCH_SIZE=`expr $CORES \* 2`

rm -rf ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_throughput*

cloudtik-run \
    --enable_jemalloc \
    --throughput_mode \
    ${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/tools/test_net.py \
    $ARGS \
    --iter-warmup 10 \
    -i 20 \
    --config-file "${MODEL_DIR}/models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" \
    TEST.IMS_PER_BATCH ${BATCH_SIZE} \
    MODEL.WEIGHT "${CHECKPOINT_DIR}/e2e_mask_rcnn_R_50_FPN_1x.pth" \
    MODEL.DEVICE cpu \
    2>&1 | tee ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_throughput.log

# For the summary of results
wait

if [[ ${PLATFORM} == "linux" ]]; then
  throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/maskrcnn_${PRECISION}_inference_throughput* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
  echo ""maskrcnn";"throughput";$1;${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
fi
