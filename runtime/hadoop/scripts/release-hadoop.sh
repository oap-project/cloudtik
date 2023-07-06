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

arch=$(uname -m)

FUSE_HDFS_BIN=fuse_dfs-${HADOOP_VERSION}-${arch}

# upload
cp ./fuse_dfs ./${FUSE_HDFS_BIN}
aws s3 cp ./${FUSE_HDFS_BIN} s3://cloudtik/downloads/hadoop/

# invalidate
aws cloudfront create-invalidation --distribution-id E3703BSG9BICN1 --paths "/downloads/hadoop/${FUSE_HDFS_BIN}"
