ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

ARG ZOOKEEPER_VERSION=3.7.1
ARG KAFKA_VERSION=3.2.3
ARG KAFKA_SCALA_VERSION=2.13
ARG PRESTO_VERSION=0.276

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

# TODO: Install prometheus

RUN wget https://downloads.apache.org/zookeeper/zookeeper-${ZOOKEEPER_VERSION}/apache-zookeeper-${ZOOKEEPER_VERSION}-bin.tar.gz -O zookeeper-${ZOOKEEPER_VERSION}.tar.gz -O zookeeper.tar.gz \
    && mkdir -p "$ZOOKEEPER_HOME" \
    && tar --extract --file zookeeper.tar.gz --directory "$ZOOKEEPER_HOME" --strip-components 1 --no-same-owner \
    && rm zookeeper.tar.gz \
    # Install Kafka
    && wget https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION}.tgz -O kafka.tgz \
    && mkdir -p "$KAFKA_HOME" \
    && tar --extract --file kafka.tgz --directory "$KAFKA_HOME" --strip-components 1 --no-same-owner \
    && rm kafka.tgz \
    # Install Presto
    && wget https://repo1.maven.org/maven2/com/facebook/presto/presto-server/${PRESTO_VERSION}/presto-server-${PRESTO_VERSION}.tar.gz -O presto-server.tar.gz \
    && mkdir -p "$PRESTO_HOME" \
    && tar --extract --file presto-server.tar.gz --directory "$PRESTO_HOME" --strip-components 1 --no-same-owner \
    && rm presto-server.tar.gz \
    && wget https://repo1.maven.org/maven2/com/facebook/presto/presto-cli/${PRESTO_VERSION}/presto-cli-${PRESTO_VERSION}-executable.jar \
    && mv presto-cli-${PRESTO_VERSION}-executable.jar $PRESTO_HOME/bin/presto \
    && chmod +x $PRESTO_HOME/bin/presto

WORKDIR /home/cloudtik/
