# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Conda builder
FROM ubuntu:20.04 as conda-builder
ENV HOME=/opt
ENV RUNTIME_PATH=${HOME}/runtime

RUN apt-get update -y && apt-get install --yes \
    wget \
    libdigest-sha-perl \
    bzip2 \
    && mkdir -p ${RUNTIME_PATH}

# Download miniconda 4.5.12, then upgrade it to 4.8.4
RUN wget --quiet --output-document miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh \
    && (echo '866ae9dff53ad0874e1d1a60b1ad1ef8  miniconda.sh' | md5sum -c) \
    && (echo 'e5e5b4cd2a918e0e96b395534222773f7241dc59d776db1b9f7fedfcb489157a  miniconda.sh' | shasum -a 256 -c) \
    # Conda must be installed at ${RUNTIME_PATH}/conda
    && /bin/bash miniconda.sh -b -p ${RUNTIME_PATH}/conda \
    && rm miniconda.sh \
    && ${RUNTIME_PATH}/conda/bin/conda install --name base conda=4.8.4

# Hadoop builder
FROM ubuntu:20.04 as hadoop-builder
ENV HOME=/opt
ENV RUNTIME_PATH=${HOME}/runtime

RUN apt-get update -y && apt-get install --yes \
    wget \
    && mkdir -p ${RUNTIME_PATH}

ARG HADOOP_VERSION=3.3.1

# Install Hadoop
ENV HADOOP_HOME ${RUNTIME_PATH}/hadoop

WORKDIR ${RUNTIME_PATH}
RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    mkdir -p "$HADOOP_HOME" && \
    tar --extract --file hadoop.tar.gz --directory "$HADOOP_HOME" --strip-components 1 --no-same-owner && \
    rm hadoop.tar.gz && \
    wget -nc -P "${HADOOP_HOME}/share/hadoop/tools/lib"  https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    for jar in 'jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' 'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar'; \
    do \
	    find "${HADOOP_HOME}"/share/hadoop/hdfs/lib -name $jar | xargs -i cp {} "${HADOOP_HOME}"/share/hadoop/tools/lib; \
    done && \
    mkdir -p "${HADOOP_HOME}"/spark-jars && \
    for jar in 'hadoop-aws-[0-9]*[0-9].jar' 'aws-java-sdk-bundle-[0-9]*[0-9].jar' 'gcs-connector-hadoop3-*.jar' 'hadoop-azure-[0-9]*[0-9].jar' 'azure-storage-[0-9]*[0-9].jar' 'wildfly-openssl-[0-9]*[0-9].Final.jar' 'jetty-util-ajax-[0-9]*[0-9].v[0-9]*[0-9].jar' 'jetty-util-[0-9]*[0-9].v[0-9]*[0-9].jar'; \
    do \
	    find "${HADOOP_HOME}"/share/hadoop/tools/lib -name $jar | xargs -i cp {} "${HADOOP_HOME}"/spark-jars; \
    done

# Spark image
FROM ubuntu:20.04
ENV HOME=/opt
ENV RUNTIME_PATH=${HOME}/runtime
ENV TZ=America/Los_Angeles

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update  -y \
  && apt-get install --yes \
    bash \
    sudo \
    coreutils \
    procps \
    apt-utils \
    curl \
    wget \
    unzip \
    maven \
    git \
    tini \
    libc6 \
    libpam-modules \
    krb5-user \
    libnss3 \
    net-tools \
  && /var/lib/dpkg/info/ca-certificates-java.postinst configure \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && mkdir -p ${RUNTIME_PATH}

WORKDIR ${RUNTIME_PATH}

# Conda
COPY --from=conda-builder ${RUNTIME_PATH}/conda ${RUNTIME_PATH}/conda

# Source conda.sh for all login shells.
RUN ln -s ${RUNTIME_PATH}/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Conda recommends using strict channel priority speed up conda operations and reduce package incompatibility problems.
# Set always_yes to avoid needing -y flags, and improve conda experience in Databricks notebooks.
RUN ${RUNTIME_PATH}/conda/bin/conda config --system --set channel_priority strict \
    && ${RUNTIME_PATH}/conda/bin/conda config --system --set always_yes True


# Install JDK
ENV JAVA_HOME            $RUNTIME_PATH/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

# JDK download links refer to https://github.com/adoptium/containers
# and https://github.com/docker-library/docs/blob/master/eclipse-temurin/README.md
RUN wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.16.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.16.1_1.tar.gz -O openjdk.tar.gz && \
    mkdir -p "$JAVA_HOME" && \
    tar --extract --file openjdk.tar.gz --directory "$JAVA_HOME" --strip-components 1 --no-same-owner && \
    rm openjdk.tar.gz

ARG SPARK_VERSION=3.2.1

ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           ${RUNTIME_PATH}/spark

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz -O spark.tgz && \
    mkdir -p "$SPARK_HOME" && \
    tar --extract --file spark.tgz --directory "$SPARK_HOME" --strip-components 1 --no-same-owner && \
    rm spark.tgz && \
    wget -nc -P "${SPARK_HOME}/jars" https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/${SPARK_VERSION}/spark-hadoop-cloud_2.12-${SPARK_VERSION}.jar && \
    wget -nc -P "${SPARK_HOME}/jars"  https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    cp ${SPARK_HOME}/kubernetes/dockerfiles/spark/entrypoint.sh ${HOME}/entrypoint.sh && \
    cp ${SPARK_HOME}/kubernetes/dockerfiles/spark/decom.sh ${HOME}/decom.sh

COPY --from=hadoop-builder ${RUNTIME_PATH}/hadoop/spark-jars ${SPARK_HOME}/jars

# run
ARG spark_uid=185

RUN set -ex && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
    rm -rf /var/cache/apt/* && \
    chmod g+w ${HOME} && \
    mkdir -p ${HOME}/logs && \
    chmod g+w ${HOME}/logs

ENV SPARK_LOG_DIR   ${HOME}/logs
ENV SPARK_CONF_DIR  ${HOME}/conf

COPY  entrypoint-nop.sh ${HOME}/entrypoint-nop.sh
COPY  spark-sql.sh ${HOME}/spark-sql.sh
COPY  spark-shell.sh ${HOME}/spark-shell.sh
COPY  spark-submit.sh ${HOME}/spark-submit.sh

RUN chmod a+x ${HOME}/entrypoint.sh && \
    chmod a+x ${HOME}/decom.sh && \
    chmod a+x ${HOME}/entrypoint-nop.sh && \
    chmod a+x ${HOME}/spark-sql.sh && \
    chmod a+x ${HOME}/spark-shell.sh && \
    chmod a+x ${HOME}/spark-submit.sh

WORKDIR ${HOME}

ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
USER ${spark_uid}
