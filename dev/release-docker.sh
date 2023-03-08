#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/set-default-vars.sh

IMAGE_TAG="nightly"
CLOUDTIK_REGION="GLOBAL"

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
    --region)
        shift
        CLOUDTIK_REGION=$1
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
        echo "Usage: release-docker.sh [ --image-tag ] [ --region ] [ --python-version ] --clean --tag-nightly --no-build --no-push"
        exit 1
    esac
    shift
done

PYTHON_TAG=${PYTHON_VERSION//./}
DOCKER_REGISTRY_PRC="registry.cn-shanghai.aliyuncs.com/"

cd $CLOUDTIK_HOME

registry_regions=('GLOBAL')
if [ "${CLOUDTIK_REGION}" == "PRC" ]; then
    registry_regions[1]='PRC'
fi

if [ $DO_CLEAN ]; then
    for registry_region in ${registry_regions[@]};
    do
        DOCKER_REGISTRY=""
        if [ "${registry_region}" == "PRC" ]; then
            DOCKER_REGISTRY=${DOCKER_REGISTRY_PRC}
        fi

        for image_name in "spark-runtime-benchmark" "spark-ml-mxnet" "spark-ml-oneapi" "spark-ml-runtime" "spark-ml-base" "spark-runtime" "cloudtik" "cloudtik-deps" "cloudtik-base"
        do
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG
            if [ "$(sudo docker images -q $image 2> /dev/null)" != "" ]; then
                sudo docker rmi $image
            fi
        done
    done
fi

if [ $TAG_NIGHTLY ]; then
    for registry_region in ${registry_regions[@]};
    do
        DOCKER_REGISTRY=""
        if [ "${registry_region}" == "PRC" ]; then
            DOCKER_REGISTRY=${DOCKER_REGISTRY_PRC}
        fi

        for image_name in "cloudtik-base" "cloudtik-deps" "cloudtik" "spark-runtime" "spark-ml-base" "spark-ml-runtime" "spark-ml-oneapi" "spark-ml-mxnet" "spark-runtime-benchmark"
        do
            image_nightly=${DOCKER_REGISTRY}cloudtik/$image_name:nightly
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG
            if [ "$(sudo docker images -q $image_nightly 2> /dev/null)" != "" ]; then
                sudo docker tag $image_nightly $image
            fi
        done
    done
fi

# Default build
if [ ! $NO_BUILD ]; then
    sudo bash ./build-docker.sh --image-tag $IMAGE_TAG --region ${CLOUDTIK_REGION} --python-version ${PYTHON_VERSION} \
        --build-spark --build-ml --build-ml-oneapi --build-spark-benchmark
fi

# Default push
if [ ! $NO_PUSH ]; then
    for registry_region in ${registry_regions[@]};
    do
        DOCKER_REGISTRY=""
        if [ "${registry_region}" == "PRC" ]; then
            DOCKER_REGISTRY=${DOCKER_REGISTRY_PRC}
        fi

        for image_name in "cloudtik" "spark-runtime" "spark-ml-runtime" "spark-ml-oneapi" "spark-runtime-benchmark"
        do
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG
            if [ "$(sudo docker images -q $image 2> /dev/null)" != "" ]; then
                sudo docker push $image
            fi
        done
    done
fi
