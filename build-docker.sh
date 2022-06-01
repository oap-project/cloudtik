#!/bin/bash
# This script is for users to build docker images locally. It is most useful for users wishing to edit the
# cloudtik-base, cloudtik-deps, or cloudtik images. 

set -x

GPU=""
BASE_IMAGE="ubuntu:focal"
WHEEL_URL="https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-0.9.0-cp37-cp37m-manylinux2014_x86_64.whl"
PYTHON_VERSION="3.7.7"
CONDA_ENV_NAME="cloudtik_py37"

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
    --build-dev)
        BUILD_DEV=YES
        ;;
    --shas-only)
        # output the SHA sum of each build. This is useful for scripting tests,
        # especially when builds of different versions are running on the same machine.
        # It also can facilitate cleanup.
        OUTPUT_SHA=YES
        ;;
    --wheel-to-use)
        # Which wheel to use. This defaults to the latest nightly on python 3.7
        echo "not implemented, just hardcode me :'("
        exit 1
        ;;
    --python-version)
        # Python version to install. e.g. 3.7.7.
        # Changing python versions may require a different wheel.
        # If not provided defaults to 3.7.7
        shift
        PYTHON_VERSION=$1
        ;;
    *)
        echo "Usage: build-docker.sh [ --gpu ] [ --base-image ] [ --no-cache-build ] [ --shas-only ] [ --build-dev ] [ --wheel-to-use ] [ --python-version ]"
        exit 1
    esac
    shift
done

WHEEL_DIR=$(mktemp -d)
wget --quiet "$WHEEL_URL" -P "$WHEEL_DIR"
WHEEL="$WHEEL_DIR/$(basename "$WHEEL_DIR"/*.whl)"
# Build cloudtik-base, cloudtik-deps, and cloudtik.
for IMAGE in "cloudtik-base"
do
    cp "$WHEEL" "docker/$IMAGE/$(basename "$WHEEL")"
    if [ $OUTPUT_SHA ]; then
        IMAGE_SHA=$(docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_VERSION="$PYTHON_VERSION" --build-arg CONDA_ENV_NAME="$CONDA_ENV_NAME" -q -t cloudtik/$IMAGE:nightly$GPU docker/$IMAGE)
        echo "cloudtik/$IMAGE:nightly$GPU SHA:$IMAGE_SHA"
    else
        docker build $NO_CACHE  --build-arg GPU="$GPU" --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_VERSION="$PYTHON_VERSION" --build-arg CONDA_ENV_NAME="$CONDA_ENV_NAME" -t cloudtik/$IMAGE:nightly$GPU docker/$IMAGE
    fi
    rm "docker/$IMAGE/$(basename "$WHEEL")"
done 

for IMAGE in "cloudtik-deps" "cloudtik"
do
    cp "$WHEEL" "docker/$IMAGE/$(basename "$WHEEL")"
    if [ $OUTPUT_SHA ]; then
        IMAGE_SHA=$(docker build $NO_CACHE --build-arg GPU="$GPU" --build-arg BASE_IMAGE="nightly" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_VERSION="$PYTHON_VERSION" -q -t cloudtik/$IMAGE:nightly$GPU docker/$IMAGE)
        echo "cloudtik/$IMAGE:nightly$GPU SHA:$IMAGE_SHA"
    else
        docker build $NO_CACHE  --build-arg GPU="$GPU" --build-arg BASE_IMAGE="nightly" --build-arg WHEEL_PATH="$(basename "$WHEEL")" --build-arg PYTHON_VERSION="$PYTHON_VERSION" -t cloudtik/$IMAGE:nightly$GPU docker/$IMAGE
    fi
    rm "docker/$IMAGE/$(basename "$WHEEL")"
done 

# Build the current source
if [ $BUILD_DEV ]; then 
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

docker build  $NO_CACHE -t cloudtik/spark-runtime:nightly runtime/spark/docker
docker build  $NO_CACHE -t cloudtik/presto-runtime:nightly runtime/presto/docker
docker build  $NO_CACHE -t cloudtik/trino-runtime:nightly runtime/trino/docker
docker build  $NO_CACHE -t cloudtik/universe-runtime:nightly runtime/universe/docker
