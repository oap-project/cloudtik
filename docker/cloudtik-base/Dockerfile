# The base Docker image installs main libraries needed to run CloudTik

# The GPU options are NVIDIA CUDA developer images.
ARG BASE_IMAGE="ubuntu:focal"
FROM ${BASE_IMAGE}
# FROM directive resets ARG
ARG BASE_IMAGE
ENV TZ=America/Los_Angeles
# TODO(ilr) $HOME seems to point to result in "" instead of "/home/cloudtik"
ENV PATH "/home/cloudtik/anaconda3/bin:$PATH"
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.7.7

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

RUN sudo apt-get update -y && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
        git \
        wget \
        cmake \
        g++ \ 
        zlib1g-dev \
        tmux \
        screen \
        rsync \
        openssh-client \
        gnupg \
        jq \
        dnsutils \
    && wget \
        --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -u -p $HOME/anaconda3 \
    && $HOME/anaconda3/bin/conda init \ 
    && echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> /home/cloudtik/.bashrc \
    && rm /tmp/miniconda.sh \
    && $HOME/anaconda3/bin/conda install -y \
        libgcc python=$PYTHON_VERSION \
    && $HOME/anaconda3/bin/conda clean -y --all \
    && $HOME/anaconda3/bin/pip install --no-cache-dir \
        flatbuffers \
        psutil \
        blist \ 
    # blist is needed for numpy
    # To avoid the following error on Jenkins:
    # AttributeError: 'numpy.ufunc' object has no attribute '__module__'
    # We install cmake temporarily to get psutil, blist
    && sudo apt-get autoremove -y cmake zlib1g-dev \
        # We keep g++ on GPU images, because uninstalling removes CUDA Devel tooling
        $(if [ "$BASE_IMAGE" = "ubuntu:focal" ]; then echo \
        g++; fi) \
    # Install kubectl
    && wget -O - -q https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
        && sudo touch /etc/apt/sources.list.d/kubernetes.list \
        && echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list \
        && sudo apt-get update \
        && sudo apt-get install kubectl \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

WORKDIR $HOME
