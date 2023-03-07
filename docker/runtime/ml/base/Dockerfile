ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

ARG CLOUDTIK_REGION=""

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            pip config unset global.index-url; \
        fi

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
    # Install machine learning and basic libraries
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt
