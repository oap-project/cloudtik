ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

ARG OAP_VERSION=1.4.0.spark32

ENV OAP_VERSION        ${OAP_VERSION}
ENV OAP_HOME           $RUNTIME_PATH/oap

# Install OAP by Conda
RUN $HOME/anaconda3/bin/conda create -p "${OAP_HOME}" -c conda-forge -c intel-bigdata -c intel -y oap=${OAP_VERSION}