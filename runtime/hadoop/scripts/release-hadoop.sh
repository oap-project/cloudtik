#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
        echo "Usage: release-hadoop.sh [ --version ]"
        exit 1
    esac
    shift
done

# HDFS Fuse
arch=$(uname -m)
HDFS_FUSE_BIN=fuse_dfs-${HADOOP_VERSION}-${arch}

# upload
cp ./hadoop-hdfs-native-client/target/main/native/fuse-dfs/fuse_dfs ./${HDFS_FUSE_BIN}
aws s3 cp ./${HDFS_FUSE_BIN} s3://cloudtik/downloads/hadoop/

# HDFS NFS
HDFS_NFS_JAR=hadoop-hdfs-nfs-${HADOOP_VERSION}.jar
aws s3 cp ./hadoop-hdfs-project/hadoop-hdfs-nfs/target/${HDFS_NFS_JAR} s3://cloudtik/downloads/hadoop/

# invalidate
aws cloudfront create-invalidation --distribution-id E3703BSG9BICN1 --paths "/downloads/hadoop/${HDFS_FUSE_BIN} /downloads/hadoop/${HDFS_NFS_JAR}"
