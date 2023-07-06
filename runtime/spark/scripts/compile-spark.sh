#!/bin/bash

CLOUDTIK_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/../../..;pwd)
SPARK_VERSION=3.3.0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --version)
        # Override for the hadoop version.
        shift
        SPARK_VERSION=$1
        ;;
    --patch)
        APPLY_PATCH=YES
        ;;
    *)
        echo "Usage: compile-spark.sh [ --version ] [ --patch ]"
        exit 1
    esac
    shift
done

rm -rf /tmp/spark
git clone https://github.com/apache/spark.git /tmp/spark
cd /tmp/spark && git checkout v${SPARK_VERSION}

if [ $APPLY_PATCH ]; then
    if [ -d "$CLOUDTIK_HOME/runtime/spark/spark-${SPARK_VERSION}" ]; then
        for patch in $CLOUDTIK_HOME/runtime/spark/spark-${SPARK_VERSION}/optimizations/*.patch; do
            git apply $patch
        done
    fi
fi

./dev/make-distribution.sh --name hadoop3 --tgz -Phadoop-3 -Dhadoop.version=3.3.1 -Phive -Phive-thriftserver -Pyarn \
 -Dmaven.wagon.http.ssl.insecure=true -Dmaven.wagon.http.ssl.allowall=true -Dmaven.wagon.http.ssl.ignore.validity.dates=true

mkdir -p $CLOUDTIK_HOME/runtime/spark/dist
cp -r /tmp/spark/spark-${SPARK_VERSION}-bin-hadoop3.tgz $CLOUDTIK_HOME/runtime/spark/dist/
