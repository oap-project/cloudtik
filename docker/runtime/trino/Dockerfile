ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

ARG TRINO_VERSION=382

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

# Ganglia
RUN sudo apt-get update -y \
    && sudo apt-get install -y ganglia-monitor \
    && wget https://d30257nes7d4fq.cloudfront.net/downloads/ganglia/modpython.so \
    && sudo cp modpython.so /usr/lib/ganglia && sudo chmod 644 /usr/lib/ganglia/modpython.so \
    && rm modpython.so \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

# Install JDK
ENV JAVA_HOME            $RUNTIME_PATH/jdk12
ENV PATH                 $JAVA_HOME/bin:$PATH

RUN wget https://download.java.net/java/GA/jdk12/33/GPL/openjdk-12_linux-x64_bin.tar.gz  && \
    gunzip openjdk-12_linux-x64_bin.tar.gz && \
    tar -xf openjdk-12_linux-x64_bin.tar && \
    rm openjdk-12_linux-x64_bin.tar && \
    mv jdk-12 jdk12

# Trino
ENV TRINO_VERSION        ${TRINO_VERSION}
ENV TRINO_HOME           $RUNTIME_PATH/trino
ENV PATH                 $TRINO_HOME/bin:$PATH

# Install Trino
RUN wget https://repo1.maven.org/maven2/io/trino/trino-server/${TRINO_VERSION}/trino-server-${TRINO_VERSION}.tar.gz && \
    tar -zxf trino-server-${TRINO_VERSION}.tar.gz && \
    mv trino-server-${TRINO_VERSION} trino && \
    rm trino-server-${TRINO_VERSION}.tar.gz

WORKDIR /home/cloudtik/
