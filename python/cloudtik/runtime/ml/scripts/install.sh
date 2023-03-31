#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    -h|--head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_tools() {
    # Install necessary tools
    which numactl > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install numactl -y > /dev/null
    which cmake > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install cmake -y > /dev/null
    which g++-9 > /dev/null || sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install g++-9 -y > /dev/null
}

function setup_oneapi_repository() {
    wget -O- -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list > /dev/null \
    && sudo apt-get update -y > /dev/null
}

function install_ml() {
    CLOUDTIK_ENV_ROOT=$(dirname $(dirname $(which cloudtik)))

    if ([ "$ML_WITH_ONEAPI" == "true" ] \
       || [ "$ML_WITH_INTEL_MPI" == "true" ] \
       ||[ "$ML_WITH_ONECCL" == "true" ] \
       || [ "$ML_WITH_INTEL_PYTORCH" == "true" ]) \
       && ([ "$ML_WITH_INTEL_MPI" != "false" ] \
       || [ "$ML_WITH_ONECCL" != "false" ] \
       || [ "$ML_WITH_INTEL_PYTORCH" != "false" ]); then
        setup_oneapi_repository
    fi

    # Install Machine Learning libraries and components
    echo "Installing machine learning tools: mlflow, hyperopt..."
    pip --no-cache-dir -qq install mlflow==2.1.1 SQLAlchemy==1.4.46 alembic==1.10.1 pyarrow==8.0.0 hyperopt==0.2.7 scikit-learn==1.0.2
    mkdir -p $RUNTIME_PATH/mlflow

    if [ "$ML_WITH_MXNET" == "true" ]; then
        echo "Installing deep learning frameworks: mxnet..."
        pip --no-cache-dir -qq install mxnet==1.9.1 gluoncv==0.10.5.post0
    else
        echo "Installing deep learning frameworks: tensorflow, pytorch..."
        pip --no-cache-dir -qq install tensorflow==2.9.3
        pip --no-cache-dir -qq install torch==1.13.1 torchvision==0.14.1 \
            --extra-index-url https://download.pytorch.org/whl/cpu

        if ([ "$ML_WITH_ONEAPI" == "true" ] || [ "$ML_WITH_INTEL_PYTORCH" == "true" ]) \
            && [ "$ML_WITH_INTEL_PYTORCH" != "false" ]; then
            # Install Jemalloc and Intel OpenMP for better performance
            conda install jemalloc intel-openmp -p ${CLOUDTIK_ENV_ROOT} -y > /dev/null
            pip --no-cache-dir -qq install intel-extension-for-pytorch==1.13.100+cpu \
                oneccl_bind_pt==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu
        fi
        pip --no-cache-dir -qq install transformers==4.11.0
    fi

    echo "Installing deep learning libraries for music and audio analysis..."
    pip --no-cache-dir -qq install librosa==0.9.2

    echo "Installing deep learning libraries for facial recognition..."
    pip --no-cache-dir -qq install opencv-python-headless==4.6.0.66
    if [ "$ML_WITH_MXNET" != "true" ]; then
        pip --no-cache-dir -qq install tensorflow-addons==0.17.1
    fi

    # Installing MPI
    if ([ "$ML_WITH_ONEAPI" == "true" ] || [ "$ML_WITH_INTEL_MPI" == "true" ]) \
        && [ "$ML_WITH_INTEL_MPI" != "false" ]; then
        echo "Installing Intel MPI..."
        ONEAPI_MPI_HOME=/opt/intel/oneapi/mpi
        if [ ! -d "${ONEAPI_MPI_HOME}" ]; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y intel-oneapi-mpi-2021.8.0 intel-oneapi-mpi-devel-2021.8.0 > /dev/null
            echo "if [ -f '/opt/intel/oneapi/mpi/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/mpi/latest/env/vars.sh'; fi" >> ~/.bashrc
        fi
        source ${ONEAPI_MPI_HOME}/latest/env/vars.sh
    else
        echo "Installing Open MPI..."
        which mpirun > /dev/null \
        || (mkdir /tmp/openmpi \
        && PREV_CUR_DIR=$(pwd) \
        && cd /tmp/openmpi \
        && wget -q --show-progress https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.4.tar.gz -O openmpi.tar.gz  \
        && tar --extract --file openmpi.tar.gz --directory /tmp/openmpi --strip-components 1 --no-same-owner \
        && echo "Open MPI: configure..." \
        && sudo ./configure --enable-orterun-prefix-by-default CC=gcc-9 CXX=g++-9 > /dev/null 2>&1 \
        && echo "Open MPI: make..." \
        && sudo make -j $(nproc) all > /dev/null 2>&1 \
        && echo "Open MPI: make install..." \
        && sudo make install > /dev/null 2>&1 \
        && sudo ldconfig \
        && cd ${PREV_CUR_DIR} \
        && sudo rm -rf /tmp/openmpi)
    fi

    if ([ "$ML_WITH_ONEAPI" == "true" ] || [ "$ML_WITH_ONECCL" == "true" ]) \
        && [ "$ML_WITH_ONECCL" != "false" ]; then
        echo "Installing oneCCL..."
        ONEAPI_COMPILER_HOME=/opt/intel/oneapi/compiler
        ONEAPI_TBB_HOME=/opt/intel/oneapi/tbb
        if [ ! -d "${ONEAPI_COMPILER_HOME}" ]; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y intel-oneapi-compiler-dpcpp-cpp-runtime-2023.0.0 intel-oneapi-compiler-shared-runtime-2023.0.0 > /dev/null
            echo "if [ -f '/opt/intel/oneapi/tbb/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/tbb/latest/env/vars.sh'; fi" >> ~/.bashrc
            echo "if [ -f '/opt/intel/oneapi/compiler/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/compiler/latest/env/vars.sh'; fi" >> ~/.bashrc
        fi
        source ${ONEAPI_TBB_HOME}/latest/env/vars.sh
        source ${ONEAPI_COMPILER_HOME}/latest/env/vars.sh

        ONEAPI_MKL_HOME=/opt/intel/oneapi/mkl
        if [ ! -d "${ONEAPI_MKL_HOME}" ]; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y intel-oneapi-mkl-2023.0.0 > /dev/null
            echo "if [ -f '/opt/intel/oneapi/mkl/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/mkl/latest/env/vars.sh'; fi" >> ~/.bashrc
        fi
        source ${ONEAPI_MKL_HOME}/latest/env/vars.sh

        ONEAPI_CCL_HOME=/opt/intel/oneapi/ccl
        if [ ! -d "${ONEAPI_CCL_HOME}" ]; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y intel-oneapi-ccl-2021.8.0 intel-oneapi-ccl-devel-2021.8.0 > /dev/null
            echo "if [ -f '/opt/intel/oneapi/ccl/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/ccl/latest/env/vars.sh'; fi" >> ~/.bashrc
        fi
        source ${ONEAPI_CCL_HOME}/latest/env/vars.sh
        # Configure Horovod to use CCL
        export HOROVOD_CPU_OPERATIONS=CCL
    fi

    echo "Installing Horovod..."
    if [ "$ML_WITH_MXNET" == "true" ]; then
        export CXX=/usr/bin/g++-9 && HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WIT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip --no-cache-dir -qq install horovod[mxnet,spark]==0.25.0
    else
        export CXX=/usr/bin/g++-9 && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip --no-cache-dir -qq install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.25.0
    fi
}

install_tools
install_ml
clean_install_cache
