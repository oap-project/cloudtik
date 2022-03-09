ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik:"$BASE_IMAGE"

# Install spark based Analytics + AI platform components
ARG HADOOP_VERSION=3.2.0
ARG SPARK_VERSION=3.1.1


ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

#hadoop
ENV HADOOP_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_YARN_HOME $RUNTIME_PATH/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV YARN_CONF_DIR $HADOOP_HOME/etc/hadoop

RUN wget http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O hadoop.tar.gz && \
    tar -zxf hadoop.tar.gz && \
    mv hadoop-${HADOOP_VERSION} hadoop && \
    rm hadoop.tar.gz

#java
ENV JAVA_HOME            /home/cloudtik/runtime/jdk
ENV PATH                 $JAVA_HOME/bin:$PATH

RUN wget https://devops.egov.org.in/Downloads/jdk/jdk-8u192-linux-x64.tar.gz  && \
    gunzip jdk-8u192-linux-x64.tar.gz && \
    tar -xf jdk-8u192-linux-x64.tar && \
    rm jdk-8u192-linux-x64.tar && \
    mv jdk1.8.0_192 jdk

#spark
ENV SPARK_VERSION        ${SPARK_VERSION}
ENV SPARK_HOME           $RUNTIME_PATH/spark
ENV PATH                 $SPARK_HOME/bin:$PATH

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz


WORKDIR /home/cloudtik/

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r  /tmp/requirements.txt

# jupyter config
RUN echo 'export PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip\
export PYSPARK_PYTHON=${CONDA_PREFIX}/bin/python \
export PYSPARK_DRIVER_PYTHON=${CONDA_PREFIX}/bin/python' >> $HOME/anaconda3/etc/jupyterrc

#access jupyter server remotely
RUN mkdir -p /home/cloudtik/jupyter && echo Y|jupyter notebook --generate-config
RUN sed -i  "1 ic.NotebookApp.ip = '*'" ~/.jupyter/jupyter_notebook_config.py && \
    sed -i  '1 ic.NotebookApp.open_browser = False' ~/.jupyter/jupyter_notebook_config.py && \
    sed -i  "1 ic.NotebookApp.notebook_dir = '/home/cloudtik/jupyter'" ~/.jupyter/jupyter_notebook_config.py && \
    # set default password(cloudtik) for jupyter lab
    sed -i  "1 ic.NotebookApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$Y+sBd6UhAyKNsI+/mHsy9g\$WzJsUujSzmotUkblSTpMwCFoOBVSwm7S5oOPzpC+tz8'" ~/.jupyter/jupyter_notebook_config.py && \
    sed -i  '1 ic.NotebookApp.allow_root = True'   ~/.jupyter/jupyter_notebook_config.py

# jupyter kernel
RUN python -m spylon_kernel install --user

