ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

ARG PRESTO_VERSION=0.271.1

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

RUN wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
    gunzip jdk-8u192-linux-x64.tar.gz && \
    tar -xf jdk-8u192-linux-x64.tar && \
    rm jdk-8u192-linux-x64.tar && \
    mv jdk1.8.0_192 jdk

# Presto
ENV PRESTO_VERSION        ${PRESTO_VERSION}
ENV PRESTO_HOME           $RUNTIME_PATH/presto
ENV PATH                  $PRESTO_HOME/bin:$PATH

# Install Presto
RUN wget https://repo1.maven.org/maven2/com/facebook/presto/presto-server/${PRESTO_VERSION}/presto-server-${PRESTO_VERSION}.tar.gz \
    && tar -zxf presto-server-${PRESTO_VERSION}.tar.gz \
    && mv presto-server-${PRESTO_VERSION} presto \
    && rm presto-server-${PRESTO_VERSION}.tar.gz \
    && wget https://repo1.maven.org/maven2/com/facebook/presto/presto-cli/${PRESTO_VERSION}/presto-cli-${PRESTO_VERSION}-executable.jar \
    && mv presto-cli-${PRESTO_VERSION}-executable.jar $PRESTO_HOME/bin/presto \
    && chmod +x $PRESTO_HOME/bin/presto

WORKDIR /home/cloudtik/
