ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ai-base:"$BASE_IMAGE"

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    # add signed entry to apt sources and configure the APT client to use Intel repository:
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && sudo apt-get update -y \
    && sudo apt-get install -y \
        intel-oneapi-mpi-2021.8.0 \
        intel-oneapi-mpi-devel-2021.8.0 \
        intel-oneapi-compiler-dpcpp-cpp-runtime-2023.0.0 \
        intel-oneapi-compiler-shared-runtime-2023.0.0 \
        intel-oneapi-ccl-2021.8.0 \
        intel-oneapi-ccl-devel-2021.8.0 \
        intel-oneapi-mkl-2023.0.0 \
    && echo "if [ -f '/opt/intel/oneapi/mpi/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/mpi/latest/env/vars.sh'; fi" >> ~/.bashrc \
    && echo "if [ -f '/opt/intel/oneapi/tbb/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/tbb/latest/env/vars.sh'; fi" >> ~/.bashrc \
    && echo "if [ -f '/opt/intel/oneapi/compiler/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/compiler/latest/env/vars.sh'; fi" >> ~/.bashrc \
    && echo "if [ -f '/opt/intel/oneapi/ccl/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/ccl/latest/env/vars.sh'; fi" >> ~/.bashrc \
    && echo "if [ -f '/opt/intel/oneapi/mkl/latest/env/vars.sh' ]; then . '/opt/intel/oneapi/mkl/latest/env/vars.sh'; fi" >> ~/.bashrc \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install deep learning libraries (Tensorflow, Keras)
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    # Install deep learning libraries (PyTorch and torch tools)
    && pip --no-cache-dir install torch==1.13.1 torchvision==0.14.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    # Install Jemalloc and intel-openmp for better performance for Intel Extension for PyTorch
    && conda install jemalloc intel-openmp -p $HOME/anaconda3/envs/$CLOUDTIK_ENV -y \
    && conda clean -itqy \
    # Install Intel Extension for PyTorch and oneCCL binding
    && pip --no-cache-dir install intel-extension-for-pytorch==1.13.100+cpu \
        oneccl_bind_pt==1.13.0+cpu -f https://developer.intel.com/ipex-whl-stable-cpu \
    # Source the mpi and ccl vars
    && . /opt/intel/oneapi/mpi/latest/env/vars.sh \
    && . /opt/intel/oneapi/tbb/latest/env/vars.sh \
    && . /opt/intel/oneapi/compiler/latest/env/vars.sh \
    && . /opt/intel/oneapi/ccl/latest/env/vars.sh \
    # Install Horovod
    && export HOROVOD_CPU_OPERATIONS=CCL \
    && export CXX=/usr/bin/g++-9 \
    && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.27.0

ENV AI_WITH_ONEAPI true
