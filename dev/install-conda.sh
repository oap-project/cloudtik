#!/usr/bin/env bash

arch=$(uname -m)
conda_download_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${arch}.sh"

wget \
        --quiet ${conda_download_url} \
        -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -u -p $HOME/anaconda3 \
    && $HOME/anaconda3/bin/conda init \
    && rm /tmp/miniconda.sh \
