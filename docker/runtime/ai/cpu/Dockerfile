ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ai-base:"$BASE_IMAGE"

WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install deep learning libraries (Tensorflow, Keras)
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    # Install deep learning libraries (PyTorch and torch tools)
    && pip --no-cache-dir install torch==1.13.1 torchvision==0.14.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    # Install OpenMPI
    && mkdir /tmp/openmpi \
    && PREV_CUR_DIR=$(pwd) \
    && cd /tmp/openmpi \
    && wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.4.tar.gz -O openmpi.tar.gz  \
    && tar --extract --file openmpi.tar.gz --directory /tmp/openmpi --strip-components 1 --no-same-owner \
    && sudo ./configure --enable-orterun-prefix-by-default CC=gcc-9 CXX=g++-9 \
    && sudo make -j $(nproc) all \
    && sudo make install \
    && sudo ldconfig \
    && cd ${PREV_CUR_DIR} \
    && sudo rm -rf /tmp/openmpi \
    # Install Horovod
    && export CXX=/usr/bin/g++-9 \
    && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip --no-cache-dir install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.27.0

ENV AI_WITH_CPU true
