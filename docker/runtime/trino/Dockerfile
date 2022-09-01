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
ENV JAVA_HOME            $RUNTIME_PATH/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

# JDK download links refer to https://github.com/adoptium/containers
# and https://github.com/docker-library/docs/blob/master/eclipse-temurin/README.md
RUN wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.16.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.16.1_1.tar.gz -O openjdk.tar.gz && \
    mkdir -p "$JAVA_HOME" && \
    tar --extract --file openjdk.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
    rm openjdk.tar.gz

# Install Trino
ENV TRINO_VERSION        ${TRINO_VERSION}
ENV TRINO_HOME           $RUNTIME_PATH/trino
ENV PATH                 $TRINO_HOME/bin:$PATH

RUN wget https://repo1.maven.org/maven2/io/trino/trino-server/${TRINO_VERSION}/trino-server-${TRINO_VERSION}.tar.gz -O trino-server.tar.gz && \
    mkdir -p "$TRINO_HOME" && \
    tar --extract --file trino-server.tar.gz --directory "$TRINO_HOME" --strip-components 1 --no-same-owner && \
    rm trino-server.tar.gz && \
    wget https://repo1.maven.org/maven2/io/trino/trino-cli/${TRINO_VERSION}/trino-cli-${TRINO_VERSION}-executable.jar && \
    mv trino-cli-${TRINO_VERSION}-executable.jar $TRINO_HOME/bin/trino && \
    chmod +x $TRINO_HOME/bin/trino

WORKDIR /home/cloudtik/
