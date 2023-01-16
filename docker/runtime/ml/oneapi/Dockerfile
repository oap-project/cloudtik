ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ml-base:"$BASE_IMAGE"

WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install deep learning libraries (Tensorflow, Keras)
    && pip install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    # Install deep learning libraries (PyTorch and torch tools)
    && pip install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    # Install IntelMPI
    && pip -qq install impi-rt==2021.8.0 impi-devel==2021.8.0 \
    && source ${CONDA_ROOT}/bin/activate ${CLOUDTIK_ENV} \
    # Install Horovod
    && export CXX=/usr/bin/g++-9 \
    && HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip install horovod[tensorflow,keras,pytorch,spark,pytorch-spark]==0.25.0
