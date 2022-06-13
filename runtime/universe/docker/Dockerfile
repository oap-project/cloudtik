ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

ARG ZOOKEEPER_VERSION=3.7.1
ARG KAFKA_VERSION=3.1.0
ARG KAFKA_SCALA_VERSION=2.13
ARG PRESTO_VERSION=0.271.1

HEALTHCHECK CMD exit 0

WORKDIR /home/cloudtik/runtime

# ZooKeeper
ENV ZOOKEEPER_VERSION    ${ZOOKEEPER_VERSION}
ENV ZOOKEEPER_HOME       $RUNTIME_PATH/zookeeper

# Kafka
ENV KAFKA_VERSION        ${KAFKA_VERSION}
ENV KAFKA_HOME           $RUNTIME_PATH/kafka

# Presto
ENV PRESTO_VERSION        ${PRESTO_VERSION}
ENV PRESTO_HOME           $RUNTIME_PATH/presto

ENV PATH                  $PRESTO_HOME/bin:$KAFKA_HOME/bin:$ZOOKEEPER_HOME/bin:$PATH

RUN sudo apt-get update -y \
    && sudo apt-get install -y ganglia-monitor \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN wget https://downloads.apache.org/zookeeper/zookeeper-${ZOOKEEPER_VERSION}/apache-zookeeper-${ZOOKEEPER_VERSION}-bin.tar.gz -O zookeeper-${ZOOKEEPER_VERSION}.tar.gz \
    && tar -zxf zookeeper-${ZOOKEEPER_VERSION}.tar.gz \
    && mv apache-zookeeper-${ZOOKEEPER_VERSION}-bin zookeeper \
    && rm zookeeper-${ZOOKEEPER_VERSION}.tar.gz \
    # Install Kafka
    && wget https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION}.tgz -O kafka-${KAFKA_VERSION}.tgz \
    && tar -zxf kafka-${KAFKA_VERSION}.tgz \
    && mv kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION} kafka \
    && rm kafka-${KAFKA_VERSION}.tgz \
    # Install Presto
    && wget https://repo1.maven.org/maven2/com/facebook/presto/presto-server/${PRESTO_VERSION}/presto-server-${PRESTO_VERSION}.tar.gz \
    && tar -zxf presto-server-${PRESTO_VERSION}.tar.gz \
    && mv presto-server-${PRESTO_VERSION} presto \
    && rm presto-server-${PRESTO_VERSION}.tar.gz \
    && wget https://repo1.maven.org/maven2/com/facebook/presto/presto-cli/${PRESTO_VERSION}/presto-cli-${PRESTO_VERSION}-executable.jar \
    && mv presto-cli-${PRESTO_VERSION}-executable.jar $PRESTO_HOME/bin/presto \
    && chmod +x $PRESTO_HOME/bin/presto

# Install python packages
WORKDIR /home/cloudtik/
COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip install -r  /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt
