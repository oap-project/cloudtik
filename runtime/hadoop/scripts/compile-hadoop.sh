#!/bin/bash

CLOUDTIK_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/../../..;pwd)
HADOOP_VERSION=3.3.1

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --version)
        # Override for the hadoop version.
        shift
        HADOOP_VERSION=$1
        ;;
    *)
        echo "Usage: compile-hadoop.sh [ --version ]"
        exit 1
    esac
    shift
done

# We are assuming to use Hadoop start-build-env.sh and running this in docker
HADOOP_SRC_DIR=$HOME/hadoop
cd $HADOOP_SRC_DIR && git checkout rel/release-${HADOOP_VERSION}

# Apply patches if we have
if [ -d "$CLOUDTIK_HOME/runtime/hadoop/hadoop-${HADOOP_VERSION}" ]; then
    for patch in $CLOUDTIK_HOME/runtime/hadoop/hadoop-${HADOOP_VERSION}/*.patch; do
        git apply $patch
    done
fi

# Create binary distribution without native code and without documentation:
mvn package -Pdist -DskipTests -Dtar -Dmaven.javadoc.skip=true

# Build fuse-dfs executable
cd hadoop-hdfs-project
mvn clean package -Pnative -pl hadoop-hdfs-native-client -am -Drequire.fuse=true -DskipTests -Dmaven.javadoc.skip=true
cp hadoop-hdfs-native-client/target/main/native/fuse-dfs/fuse_dfs ${HADOOP_SRC_DIR}
