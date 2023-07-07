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
    --patch)
        APPLY_PATCH=YES
        ;;
    --clean)
        CLEAN_BUILD=YES
        ;;
    --no-dist)
        NO_BUILD_DIST=YES
        ;;
    --no-fuse)
        NO_BUILD_FUSE=YES
        ;;
    *)
        echo "Usage: compile-hadoop.sh [ --version ] [ --patch ] [ --clean ] [ --no-dist ] [ --no-fuse ]"
        exit 1
    esac
    shift
done

# We are assuming to use Hadoop start-build-env.sh and running this in docker
HADOOP_SRC_DIR=$HOME/hadoop
cd $HADOOP_SRC_DIR && git checkout rel/release-${HADOOP_VERSION}

if [ $APPLY_PATCH ]; then
    # Apply patches if we have
    if [ -d "$CLOUDTIK_HOME/runtime/hadoop/hadoop-${HADOOP_VERSION}" ]; then
        for patch in $CLOUDTIK_HOME/runtime/hadoop/hadoop-${HADOOP_VERSION}/*.patch; do
            git apply $patch
        done
    fi
fi

CLEAN_CMD=""
if [ $CLEAN_BUILD ]; then
    CLEAN_CMD="clean"
fi

if [ ! $NO_BUILD_DIST ]; then
    # Create binary distribution without native code and without documentation:
    mvn $CLEAN_CMD package -Pdist -DskipTests -Dtar -Dmaven.javadoc.skip=true
fi

if [ ! $NO_BUILD_FUSE ]; then
    # Build fuse-dfs executable
    cd hadoop-hdfs-project
    mvn $CLEAN_CMD package -Pnative -pl hadoop-hdfs-native-client -am -Drequire.fuse=true -DskipTests -Dmaven.javadoc.skip=true
fi
