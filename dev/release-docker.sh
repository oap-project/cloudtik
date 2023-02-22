#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/set-default-vars.sh

IMAGE_TAG="nightly"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --image-tag)
        # Override for the image tag.
        shift
        IMAGE_TAG=$1
        ;;
    --python-version)
        # Python version
        shift
        PYTHON_VERSION=$1
        ;;
    --clean)
        # Remove the local images for the image tag.
        DO_CLEAN=YES
        ;;
    --tag-nightly)
        # Tag nightly to the specified image tag
        TAG_NIGHTLY=YES
        ;;
    --no-build)
        # Build with the image tag
        NO_BUILD=YES
        ;;
    --no-push)
        # Do build only, no push
        NO_PUSH=YES
        ;;
    *)
        echo "Usage: release-docker.sh [ --image-tag ] [ --python-version ] --clean --tag-nightly --no-build --no-push"
        exit 1
    esac
    shift
done

PYTHON_TAG=${PYTHON_VERSION//./}

cd $CLOUDTIK_HOME

if [ $DO_CLEAN ]; then
    sudo docker rmi cloudtik/spark-runtime-benchmark:$IMAGE_TAG
    sudo docker rmi cloudtik/spark-ml-oneapi:$IMAGE_TAG
    sudo docker rmi cloudtik/spark-ml-mxnet:$IMAGE_TAG
    sudo docker rmi cloudtik/spark-ml-runtime:$IMAGE_TAG
    sudo docker rmi cloudtik/spark-ml-base:$IMAGE_TAG
    sudo docker rmi cloudtik/spark-runtime:$IMAGE_TAG
    sudo docker rmi cloudtik/cloudtik:$IMAGE_TAG
    sudo docker rmi cloudtik/cloudtik-deps:$IMAGE_TAG
    sudo docker rmi cloudtik/cloudtik-base:$IMAGE_TAG
fi

if [ $TAG_NIGHTLY ]; then
    sudo docker tag cloudtik/spark-runtime-benchmark:nightly cloudtik/spark-runtime-benchmark:$IMAGE_TAG
    sudo docker tag cloudtik/spark-ml-oneapi:nightly cloudtik/spark-ml-oneapi:$IMAGE_TAG
    sudo docker tag cloudtik/spark-ml-mxnet:nightly cloudtik/spark-ml-mxnet:$IMAGE_TAG
    sudo docker tag cloudtik/spark-ml-runtime:nightly cloudtik/spark-ml-runtime:$IMAGE_TAG
    sudo docker tag cloudtik/spark-ml-base:nightly cloudtik/spark-ml-base:$IMAGE_TAG
    sudo docker tag cloudtik/spark-runtime:nightly cloudtik/spark-runtime:$IMAGE_TAG
    sudo docker tag cloudtik/cloudtik:nightly cloudtik/cloudtik:$IMAGE_TAG
    sudo docker tag cloudtik/cloudtik-deps:nightly cloudtik/cloudtik-deps:$IMAGE_TAG
    sudo docker tag cloudtik/cloudtik-base:nightly cloudtik/cloudtik-base:$IMAGE_TAG
fi

# Default build
if [ ! $NO_BUILD ]; then
    sudo bash ./build-docker.sh --image-tag $IMAGE_TAG --python-version ${PYTHON_VERSION} \
        --build-spark --build-ml --build-ml-mxnet --build-ml-oneapi --build-spark-benchmark
fi

# Default push
if [ ! $NO_PUSH ]; then
    sudo docker push cloudtik/cloudtik:$IMAGE_TAG
    sudo docker push cloudtik/spark-runtime:$IMAGE_TAG
    sudo docker push cloudtik/spark-ml-runtime:$IMAGE_TAG
    sudo docker push cloudtik/spark-ml-mxnet:$IMAGE_TAG
    sudo docker push cloudtik/spark-ml-oneapi:$IMAGE_TAG
    sudo docker push cloudtik/spark-runtime-benchmark:$IMAGE_TAG
fi
