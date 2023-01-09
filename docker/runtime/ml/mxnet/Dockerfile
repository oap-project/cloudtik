ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime-base:"$BASE_IMAGE"

WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install deep learning libraries (MXNet)
    && pip install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt \
    # Install Horovod
    && HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 pip install horovod[mxnet,spark]==0.25.0
