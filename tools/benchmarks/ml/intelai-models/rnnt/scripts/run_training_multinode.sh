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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/../../common/scripts/setenv.sh

RNNT_HOME=$INTELAI_MODELS_WORKSPACE/rnn-t

RNNT_OUTPUT_DIR=$RNNT_HOME/output
RNNT_DATASET_DIR=$RNNT_HOME/data
RNNT_CHECKPOINT_DIR=$RNNT_HOME/checkpoint

# Env vars
export OUTPUT_DIR=$RNNT_OUTPUT_DIR
export DATASET_DIR=$RNNT_DATASET_DIR
export CHECKPOINT_DIR=$RNNT_CHECKPOINT_DIR

mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

#(fp32, bf16, bf32)
PRECISION=fp32
NUM_STEPS=100
BACKEND=gloo

HOSTS=$(cloudtik head worker-ips --separator ",")

function usage(){
    echo "Usage: run-training_multinode.sh  [--precision fp32 | bf16 | bf32]  [--num-steps 100]  [ --backend ccl | gloo]"
    exit 1
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --precision)
        # training or inference
        shift
        PRECISION=$1
        ;;
    --num-steps)
        # num for steps
        shift
        NUM_STEPS=$1
        ;;
    --backend)
        shift
        BACKEND=$1
        ;;
    *)
        usage
    esac
    shift
done

if [ ! -e "${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at the: \${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train.py"
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


MODEL_CONFIG=${5:-"${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/configs/rnnt.toml"}
RESULT_DIR=${6:-"${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/results"}
CHECKPOINT=${7:-"none"}
CREATE_LOGFILE=${8:-"true"}
CUDNN_BENCHMARK=${9:-"true"}
NUM_GPUS=${10:-0}
EPOCHS=${12:-1}
SEED=${13:-2021}
BATCH_SIZE=${14:-32}
EVAL_BATCH_SIZE=${15:-2}
LEARNING_RATE=${16:-"0.001"}
LEARNING_RATE_WARMUP=${17:-"8000"}
GRADIENT_ACCUMULATION_STEPS=${18:-1}
LAUNCH_OPT=${LAUNCH_OPT:-"none"}
SOCKETS=$(cloudtik head info --sockets-per-worker)
NNODES=$(cloudtik head info --total-workers)
HOSTFILE=${HOSTFILE:-"${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/training/cpu/hostfile"}
NUM_RANKS=$(( NNODES * SOCKETS ))

mkdir -p $RESULT_DIR

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

PREC=""
if [ "$PRECISION" = "bf16" ]; then
    PREC="--bf16"
    precision="bf16"
    echo "### running bf16 datatype"
elif [ "$PRECISION" = "fp32" ] ; then
    PREC="--fp32"
    precision="fp32"
    echo "### running fp32 datatype"
elif [ "$PRECISION" = "bf32" ] ; then
    PREC="--bf32"
    precision="bf32"
    echo "### running bf32 datatype"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions now are: fp32, bf16 and bf32"
fi

HOSTS=$(cloudtik head worker-ips --separator ",")

IPEX="--ipex"

PROFILE=""
if [ "$3" = profiling ]; then
    PROFILE="--profiling"
fi

WARMUP=20

if [ "$CHECKPOINT" = "none" ] ; then
   CHECKPOINT=""
else
   CHECKPOINT=" --ckpt=${CHECKPOINT}"
fi

CMD=" --batch_size=$BATCH_SIZE"
CMD+=" --eval_batch_size=$EVAL_BATCH_SIZE"
CMD+=" --num_epochs=$EPOCHS"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --lr_warmup=$LEARNING_RATE_WARMUP"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=adam"
CMD+=" --dataset_dir=$DATASET_DIR/dataset/LibriSpeech"
CMD+=" --val_manifest=$DATASET_DIR/dataset/LibriSpeech/librispeech-dev-clean-wav.json"
CMD+=" --train_manifest=$DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-100-wav.json,$DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-360-wav.json,$DATASET_DIR/dataset/LibriSpeech/librispeech-train-other-500-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=100"
CMD+=" --eval_freq=1"
CMD+=" --train_freq=5"
CMD+=" --lr_decay"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
CMD+=" $CHECKPOINT"
CMD+=" $PREC"
CMD+=" $IPEX"
CMD+=" --warmup=$WARMUP"
CMD+=" $PROFILE"
CMD+=" --backend=$BACKEND"
# TODO: FP32 is still under development. For current validation,
# in FP32, it only runs 100 iterations. NUM_STEPS is disabled in FP32.
if [ "$PRECISION" = "fp32" ] ; then
    CMD+=" --num_steps=100"
elif [[ ! -z "${NUM_STEPS}" ]]; then
    CMD+=" --num_steps=$NUM_STEPS"
fi

rm -rf ${OUTPUT_DIR}/distributed_throughput_log*

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

python -m cloudtik-ml-run \
    --distributed \
    --hosts ${HOSTS} \
    --log_path=${OUTPUT_DIR} \
    --log_file_prefix="./distributed_throughput_log_${precision}" \
    ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/training/cpu/train.py \
    $CMD 2>&1 | tee ${OUTPUT_DIR}/distributed_throughput_log_${precision}.txt

# For the summary of results
wait
throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/distributed_throughput_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
echo ""RNN-T";"training distributed throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
