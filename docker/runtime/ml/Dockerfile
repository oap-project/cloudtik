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
    && pip install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    && pip install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    && conda install dlib=19.24.0 libffi=3.3 -p $HOME/anaconda3/envs/$CLOUDTIK_ENV -c conda-forge -y \
    && export CXX=/usr/bin/g++-9 \
    && HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1  pip install horovod[all-frameworks]==0.25.0
