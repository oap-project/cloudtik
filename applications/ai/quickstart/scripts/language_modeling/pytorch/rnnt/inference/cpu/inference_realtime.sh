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

if [ ! -e "${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/results/rnnt.pt" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/results/rnnt.pt does not exist"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/dataset/LibriSpeech" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/dataset/LibriSpeech does not exist"
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
if [ "$1" == "bf16" ]; then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
elif [ "$1" == "bf32" ]; then
    ARGS="$ARGS --bf32"
    echo "### running bf32 datatype"
else
    echo "### running fp32 datatype"
fi

if [[ "$USE_IPEX" == "true" ]]; then
  ARGS="$ARGS --ipex --jit"
  export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
fi

BATCH_SIZE=1
PRECISION=$1

rm -rf ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_realtime*

cloudtik-run \
    --memory-allocator=default \
    --latency_mode \
    --log-dir ${OUTPUT_DIR} \
    --log-file-prefix rnnt_${PRECISION}_inference_realtime \
    ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py \
    --dataset_dir ${DATASET_DIR}/dataset/LibriSpeech/ \
    --val_manifest ${DATASET_DIR}/dataset/LibriSpeech/librispeech-dev-clean-wav.json \
    --model_toml ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/configs/rnnt.toml \
    --ckpt ${CHECKPOINT_DIR}/results/rnnt.pt \
    --batch_size $BATCH_SIZE \
    --warm_up 10 \
    $ARGS

# For the summary of results
wait

CORES=`lscpu | grep Core | awk '{print $4}'`
CORES_PER_INSTANCE=4

INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET=`expr $CORES / $CORES_PER_INSTANCE`

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_realtime* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
BEGIN {
        sum = 0;
i = 0;
      }
      {
        sum = sum + $1;
i++;
      }
END   {
sum = sum / i * INSTANCES_PER_SOCKET;
        printf("%.2f", sum);
}')
p99_latency=$(grep 'P99 Latency' ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_realtime* |sed -e 's/.*P99 Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_THROUGHPUT_BENCHMARK_PER_SOCKET '
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
    printf("%.3f ms", sum);
}')
echo ""RNN-T";"latency";$1; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
echo ""RNN-T";"p99_latency";$1; ${BATCH_SIZE};${p99_latency}" | tee -a ${OUTPUT_DIR}/summary.log
