#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
mkdir -p $RUNTIME_PATH

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_tools() {
    # Install necessary tools
    which numactl > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install numactl -y > /dev/null)
    which cmake > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install cmake -y > /dev/null)
    which g++-9 > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install g++-9 -y > /dev/null)
    if [ "$IS_HEAD_NODE" == "true" ]; then
        which mysql > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install mysql-client -y > /dev/null)

        POSTGRES_DRIVER=$(pip freeze | grep psycopg2)
        if [ "${POSTGRES_DRIVER}" == "" ]; then
            sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install libpq-dev -y > /dev/null
        fi

        which psql > /dev/null || (sudo apt-get -qq update -y > /dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install postgresql-client -y > /dev/null)
    fi
}

function setup_oneapi_repository() {
    wget -O- -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list > /dev/null \
    && sudo apt-get update -y > /dev/null
}

function cleanup_oneapi_repository() {
    sudo rm -f /etc/apt/sources.list.d/oneAPI.list
}

function install_ai() {
    CLOUDTIK_ENV_ROOT=$(dirname $(dirname $(which cloudtik)))

    if ([ "$AI_WITH_ONEAPI" == "true" ] \
       || [ "$AI_WITH_INTEL_MPI" == "true" ] \
       || [ "$AI_WITH_ONECCL" == "true" ] \
       || [ "$AI_WITH_INTEL_PYTORCH" == "true" ]) \
       && ([ "$AI_WITH_INTEL_MPI" != "false" ] \
       || [ "$AI_WITH_ONECCL" != "false" ] \
       || [ "$AI_WITH_INTEL_PYTORCH" != "false" ]); then
        setup_oneapi_repository
    fi

    # Install Machine Learning libraries and components
    echo "Installing machine learning tools and libraries..."
    # chardet==3.0.4 from azure-cli
    pip --no-cache-dir -qq install \
        mlflow==2.3.1 \
        SQLAlchemy==1.4.46 \
        alembic==1.10.1 \
        pymysql==1.0.3 \
        pyarrow==8.0.0 \
        hyperopt==0.2.7 \
        scikit-learn==1.0.2 \
        xgboost==1.7.5 \
        transformers==4.30.2 \
        pandas==2.0.1 \
        category-encoders==2.6.0 \
        h5py==3.8.0 \
        lightgbm==3.3.5 \
        tensorflow-text==2.12.1 \
        datasets~=2.9.0 \
        tensorflow-datasets~=4.8.2 \
        tensorflow-hub~=0.12.0 \
        protobuf==3.20.3 \
        psycopg2==2.9.6

    mkdir -p $RUNTIME_PATH/mlflow

    echo "Installing deep learning frameworks and libraries..."
    pip --no-cache-dir -qq install tensorflow==2.12.0

    if [ "$AI_WITH_GPU" == "true" ]; then
        pip --no-cache-dir -qq install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
            --extra-index-url https://download.pytorch.org/whl/cu117
    else
        pip --no-cache-dir -qq install torch==1.13.1 torchvision==0.14.1 \
            --extra-index-url https://download.pytorch.org/whl/cpu
    fi

    if ([ "$AI_WITH_ONEAPI" == "true" ] || [ "$AI_WITH_INTEL_PYTORCH" == "true" ]) \
        && [ "$AI_WITH_INTEL_PYTORCH" != "false" ]; then
        # Install Jemalloc and Intel OpenMP for better performance
        conda install jemalloc intel-openmp -p ${CLOUDTIK_ENV_ROOT} -y > /dev/null
        pip --no-cache-dir -qq install intel-extension-for-pytorch==1.13.100+cpu \
            oneccl_bind_pt==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu
    fi

    pip --no-cache-dir -qq install transformers==4.11.0
    pip --no-cache-dir -qq install librosa==0.9.2 opencv-python-headless==4.6.0.66 tensorflow-addons==0.17.1

    # Installing MPI
    if ([ "$AI_WITH_ONEAPI" == "true" ] || [ "$AI_WITH_INTEL_MPI" == "true" ]) \
        && [ "$AI_WITH_INTEL_MPI" != "false" ]; then
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
        || (mkdir -p /tmp/openmpi \
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

    if ([ "$AI_WITH_ONEAPI" == "true" ] || [ "$AI_WITH_ONECCL" == "true" ]) \
        && [ "$AI_WITH_ONECCL" != "false" ]; then
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
    export CXX=/usr/bin/g++-9 && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip --no-cache-dir -qq install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.27.0

    cleanup_oneapi_repository
}

set_head_option "$@"
install_tools
install_ai
clean_install_cache
