#!/bin/bash

CLOUDTIK_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/../..;pwd)

rm -rf /tmp/spark
git clone https://github.com/apache/spark.git /tmp/spark
cd /tmp/spark && git checkout v3.3.0

git apply $CLOUDTIK_HOME/runtime/spark/optimization/0001-Top-N.patch
git apply $CLOUDTIK_HOME/runtime/spark/optimization/0002-runtime-Filter.patch

./dev/make-distribution.sh --name hadoop3  --tgz  -Phadoop-3 -Dhadoop.version=3.3.1  -Phive -Phive-thriftserver  -Pyarn \
 -Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true -Dmaven.wagon.http.ssl.ignore.validity.dates=true

mkdir -p $CLOUDTIK_HOME/runtime/spark/dist
cp -r /tmp/spark/spark-3.3.0-bin-hadoop3.tgz $CLOUDTIK_HOME/runtime/spark/dist/
