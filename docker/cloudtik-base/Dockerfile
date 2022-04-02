# The base Docker image installs main libraries needed to run CloudTik

# The GPU options are NVIDIA CUDA developer images.
ARG BASE_IMAGE="ubuntu:focal"
FROM ${BASE_IMAGE}
# FROM directive resets ARG
ARG BASE_IMAGE
ENV TZ=America/Los_Angeles
ENV PATH "/home/cloudtik/anaconda3/bin:$PATH"
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.7.7
ARG CONDA_ENV_NAME=cloudtik_py37
ARG CLOUDTIK_UID=1000
ARG CLOUDTIK_GID=100

RUN apt-get update -y \
    && apt-get install -y sudo tzdata \
    && useradd -ms /bin/bash -d /home/cloudtik cloudtik --uid $CLOUDTIK_UID --gid $CLOUDTIK_GID \
    && usermod -aG sudo cloudtik \
    && echo 'cloudtik ALL=NOPASSWD: ALL' >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

USER $CLOUDTIK_UID
ENV HOME=/home/cloudtik
ENV CLOUDTIK_ENV=$CONDA_ENV_NAME

RUN sudo apt-get update -y && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
        git \
        wget \
        tmux \
        screen \
        rsync \
        openssh-client \
        gnupg \
        jq \
        dnsutils \
    # Install Conda
    && wget \
        --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -u -p $HOME/anaconda3 \
    && $HOME/anaconda3/bin/conda init \ 
    && echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> $HOME/.bashrc \
    && rm /tmp/miniconda.sh \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN $HOME/anaconda3/bin/conda create -n $CLOUDTIK_ENV -y python=$PYTHON_VERSION \
    && export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH"

WORKDIR $HOME
