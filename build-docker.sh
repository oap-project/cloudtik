#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/dev/set-default-vars.sh

# This script is for users to build docker images locally. It is most useful for users wishing to edit the
# cloudtik-base, cloudtik-deps, or cloudtik images. 

set -x

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)
CONDA_ENV_NAME="cloudtik"
GPU=""
BASE_IMAGE="ubuntu:focal"
IMAGE_TAG="nightly"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --gpu)
        GPU="-gpu"
        BASE_IMAGE="nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04"
        ;;
    --base-image)
        # Override for the base image.
        shift
        BASE_IMAGE=$1
        ;;
    --no-cache-build)
        NO_CACHE="--no-cache"
        ;;
    --shas-only)
        # output the SHA sum of each build. This is useful for scripting tests,
        # especially when builds of different versions are running on the same machine.
        # It also can facilitate cleanup.
        OUTPUT_SHA=YES
        ;;
    --wheel-to-use)
        # Which wheel to use. This defaults to the latest nightly on minimum supported python version of CloudTik
        echo "not implemented, just hardcode me :'("
        exit 1
        ;;
    --python-version)
        # Python version
        shift
        PYTHON_VERSION=$1
        ;;
    --python-release)
        # Python release to install.
        # Changing python versions may require a different wheel.
        shift
        PYTHON_RELEASE=$1
        ;;
    --image-tag)
        shift
        IMAGE_TAG=$1
        ;;
    --build-all)
        BUILD_ALL=YES
        ;;
    --build-dev)
        BUILD_DEV=YES
        ;;
    --build-spark)
        BUILD_SPARK=YES
        ;;
    --build-spark-native-sql)
        BUILD_SPARK_NATIVE_SQL=YES
        ;;
    --build-spark-optimized)
        BUILD_SPARK_OPTIMIZED=YES
        ;;
    --build-universe)
        BUILD_UNIVERSE=YES
        ;;
    --build-presto)
        BUILD_PRESTO=YES
        ;;
    --build-trino)
        BUILD_TRINO=YES
        ;;
    --build-ml)
        BUILD_ML=YES
        ;;
    --build-ml-mxnet)
        BUILD_ML_MXNET=YES
        ;;
    --build-ml-oneapi)
        BUILD_ML_ONEAPI=YES
        ;;
    --build-ml-benchmark)
        BUILD_ML_BENCHMARK=YES
        ;;
    --build-spark-benchmark)
        BUILD_SPARK_BENCHMARK=YES
        ;;
    --build-spark-native-sql-benchmark)
        BUILD_SPARK_NATIVE_SQL_BENCHMARK=YES
        ;;
    --build-spark-optimized-benchmark)
        BUILD_SPARK_OPTIMIZED_BENCHMARK=YES
        ;;
    *)
        echo "Usage: build-docker.sh [ --base-image ] [ --no-cache-build ] [ --shas-only ] [ --wheel-to-use ] [ --python-version ] [ --python-release ] [ --image-tag ]"
        echo "Images to build options:"
        echo "[ --build-all ] [ --build-dev ] [ --build-spark ] [ --build-optimized ] [ --build-spark-native-sql ]"
        echo "[ --build-ml ] [ --build-ml-mxnet ] [ --build-ml-oneapi ]"
        echo "[ --build-universe ] [ --build-presto ] [ --build-trino ]"
        echo "[ --build-spark-benchmark ] [ --build-optimized-benchmark ] [ --build-spark-native-sql-benchmark ]"
        exit 1
    esac
    shift
done

PYTHON_TAG=${PYTHON_VERSION//./}

if [ "$IMAGE_TAG" == "nightly" ]; then
    WHEEL_URL="https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_x86_64.nightly.whl"
else
    WHEEL_URL="https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_x86_64.whl"
fi

WHEEL_DIR=$(mktemp -d)
wget --quiet "$WHEEL_URL" -P "$WHEEL_DIR"
WHEEL="$WHEEL_DIR/$(basename "$WHEEL_DIR"/*.whl)"
# Build cloudtik-base, cloudtik-deps, and cloudtik.
for IMAGE in "cloudtik-base"
do
    cp "$WHEEL" "docker/$IMAGE/$(basename "$WHEEL")"
    if [ $OUTPUT_SHA ]; then
        IMAGE_SHA=$(docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_RELEASE="$PYTHON_RELEASE" --build-arg CONDA_ENV_NAME="$CONDA_ENV_NAME" -q -t cloudtik/$IMAGE:$IMAGE_TAG$GPU docker/$IMAGE)
        echo "cloudtik/$IMAGE:$IMAGE_TAG$GPU SHA:$IMAGE_SHA"
    else
        docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_RELEASE="$PYTHON_RELEASE" --build-arg CONDA_ENV_NAME="$CONDA_ENV_NAME" -t cloudtik/$IMAGE:$IMAGE_TAG$GPU docker/$IMAGE
    fi
    rm "docker/$IMAGE/$(basename "$WHEEL")"
done 

for IMAGE in "cloudtik-deps" "cloudtik"
do
    cp "$WHEEL" "docker/$IMAGE/$(basename "$WHEEL")"
    if [ $OUTPUT_SHA ]; then
        IMAGE_SHA=$(docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE=$IMAGE_TAG --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_RELEASE="$PYTHON_RELEASE" -q -t cloudtik/$IMAGE:$IMAGE_TAG$GPU docker/$IMAGE)
        echo "cloudtik/$IMAGE:$IMAGE_TAG$GPU SHA:$IMAGE_SHA"
    else
        docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE=$IMAGE_TAG --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_RELEASE="$PYTHON_RELEASE" -t cloudtik/$IMAGE:$IMAGE_TAG$GPU docker/$IMAGE
    fi
    rm "docker/$IMAGE/$(basename "$WHEEL")"
done 

# Build the current source
if [ $BUILD_DEV ] || [ $BUILD_ALL ]; then
    git rev-parse HEAD > ./docker/cloudtik-dev/git-rev
    git archive -o ./docker/cloudtik-dev/cloudtik.tar "$(git rev-parse HEAD)"
    if [ $OUTPUT_SHA ]; then
        IMAGE_SHA=$(docker build --no-cache -q -t cloudtik/cloudtik-dev docker/cloudtik-dev)
        echo "cloudtik/cloudtik-dev:latest SHA:$IMAGE_SHA"
    else
        docker build --no-cache -t cloudtik/cloudtik-dev docker/cloudtik-dev
    fi
    rm ./docker/cloudtik-dev/cloudtik.tar ./docker/cloudtik-dev/git-rev
fi

rm -rf "$WHEEL_DIR"

if [ $BUILD_SPARK ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-runtime:$IMAGE_TAG docker/runtime/spark
fi

if [ $BUILD_SPARK_NATIVE_SQL ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-native-sql:$IMAGE_TAG docker/runtime/spark/native-sql
fi

if [ $BUILD_SPARK_OPTIMIZED ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-optimized:$IMAGE_TAG docker/runtime/spark/optimized
fi

if [ $BUILD_UNIVERSE ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/universe-runtime:$IMAGE_TAG docker/runtime/universe
fi

if [ $BUILD_PRESTO ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/presto-runtime:$IMAGE_TAG docker/runtime/presto
fi

if [ $BUILD_TRINO ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/trino-runtime:$IMAGE_TAG docker/runtime/trino
fi

# Build the ML base image which is needed as the base image for all other ML image
if [ $BUILD_ML ] || [ $BUILD_ML_MXNET ] || [ $BUILD_ML_ONEAPI ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-ml-base:$IMAGE_TAG docker/runtime/ml/base
fi

if [ $BUILD_ML ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-ml-runtime:$IMAGE_TAG docker/runtime/ml
fi

if [ $BUILD_ML_MXNET ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-ml-mxnet:$IMAGE_TAG docker/runtime/ml/mxnet
fi

if [ $BUILD_ML_ONEAPI ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-ml-oneapi:$IMAGE_TAG docker/runtime/ml/oneapi
fi

if [ $BUILD_ML_BENCHMARK ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-ml-runtime-benchmark:$IMAGE_TAG docker/runtime/ml/benchmark
fi

if [ $BUILD_SPARK_BENCHMARK ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-runtime-benchmark:$IMAGE_TAG docker/runtime/spark/benchmark
fi

if [ $BUILD_SPARK_NATIVE_SQL_BENCHMARK ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-native-sql-benchmark:$IMAGE_TAG docker/runtime/spark/benchmark/native-sql
fi

if [ $BUILD_SPARK_OPTIMIZED_BENCHMARK ] || [ $BUILD_ALL ]; then
    docker build $NO_CACHE --build-arg BASE_IMAGE=$IMAGE_TAG -t cloudtik/spark-optimized-benchmark:$IMAGE_TAG docker/runtime/spark/benchmark/optimized
fi
