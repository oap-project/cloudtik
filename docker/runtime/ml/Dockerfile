ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

RUN sudo apt-get update -y \
    && sudo apt-get install -y \
        cmake \
        g++-9 \
    # Clean up
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install deep learning libraries (Tensorflow, PyTorch, Keras)
    && pip install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    # Install deep learning libraries for facial recognition
    && pip install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    && conda install dlib=19.24.0 libffi=3.3 -p $HOME/anaconda3/envs/$CLOUDTIK_ENV -c conda-forge -y \
    # Install OpenMPI
    && export CXX=/usr/bin/g++-9 \
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
    && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.25.0
