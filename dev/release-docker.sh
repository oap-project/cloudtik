#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/set-default-vars.sh

GPU=""
IMAGE_TAG="nightly"
CLOUDTIK_REGION="GLOBAL"

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --gpu)
        GPU="-gpu"
        ;;
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
    --release-all)
        RELEASE_ALL=YES
        ;;
    --release-cloudtik)
        RELEASE_CLOUDTIK=YES
        ;;
    --release-spark)
        RELEASE_SPARK=YES
        ;;
    --release-ai-base)
        RELEASE_AI_BASE=YES
        ;;
    --release-ai)
        RELEASE_AI=YES
        ;;
    --release-ai-oneapi)
        RELEASE_AI_ONEAPI=YES
        ;;
    *)
        echo "Usage: release-docker.sh [ --gpu ] [ --image-tag ] [ --region ] [ --python-version ] --clean --tag-nightly --no-build --no-push"
        echo "Images to release options:"
        echo "[ --release-all ] [ --release-cloudtik ] [ --release-spark ]"
        echo "[ --release-ai-base ] [ --release-ai ] [ --release-ai-oneapi ]"
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

        CLEAN_IMAGE_NAMES=()
        if [ $RELEASE_AI_ONEAPI ] || [ $RELEASE_ALL ]; then
            CLEAN_IMAGE_NAMES+=("spark-ai-oneapi")
        fi
        if [ $RELEASE_AI ] || [ $RELEASE_ALL ]; then
            CLEAN_IMAGE_NAMES+=("spark-ai-runtime")
        fi
        if [ $RELEASE_AI_BASE ] || [ $RELEASE_ALL ]; then
            CLEAN_IMAGE_NAMES+=("spark-ai-base")
        fi
        if [ $RELEASE_SPARK ] || [ $RELEASE_ALL ]; then
            CLEAN_IMAGE_NAMES+=("spark-runtime-benchmark" "spark-runtime")
        fi
        if [ $RELEASE_CLOUDTIK ] || [ $RELEASE_ALL ]; then
            CLEAN_IMAGE_NAMES+=("cloudtik" "cloudtik-deps" "cloudtik-base")
        fi

        for image_name in ${CLEAN_IMAGE_NAMES[@]}
        do
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG$GPU
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

        TAG_IMAGE_NAMES=()
        if [ $RELEASE_CLOUDTIK ] || [ $RELEASE_ALL ]; then
            TAG_IMAGE_NAMES+=("cloudtik-base" "cloudtik-deps" "cloudtik")
        fi
        if [ $RELEASE_SPARK ] || [ $RELEASE_ALL ]; then
            TAG_IMAGE_NAMES+=("spark-runtime" "spark-runtime-benchmark")
        fi
        if [ $RELEASE_AI_BASE ] || [ $RELEASE_ALL ]; then
            TAG_IMAGE_NAMES+=("spark-ai-base")
        fi
        if [ $RELEASE_AI ] || [ $RELEASE_ALL ]; then
            TAG_IMAGE_NAMES+=("spark-ai-runtime")
        fi
        if [ $RELEASE_AI_ONEAPI ] || [ $RELEASE_ALL ]; then
            TAG_IMAGE_NAMES+=("spark-ai-oneapi")
        fi

        for image_name in ${TAG_IMAGE_NAMES[@]}
        do
            image_nightly=${DOCKER_REGISTRY}cloudtik/$image_name:nightly$GPU
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG$GPU
            if [ "$(sudo docker images -q $image_nightly 2> /dev/null)" != "" ]; then
                sudo docker tag $image_nightly $image
            fi
        done
    done
fi

# Default build
if [ ! $NO_BUILD ]; then
    BUILD_FLAGS=""
    if [ "$GPU" != "" ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --gpu"
    fi
    if [ $RELEASE_CLOUDTIK ] || [ $RELEASE_ALL ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --build-cloudtik"
    fi
    if [ $RELEASE_SPARK ] || [ $RELEASE_ALL ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --build-spark --build-spark-benchmark"
    fi
    if [ $RELEASE_AI_BASE ] || [ $RELEASE_ALL ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --build-ai-base"
    fi
    if [ $RELEASE_AI ] || [ $RELEASE_ALL ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --build-ai"
    fi
    if [ $RELEASE_AI_ONEAPI ] || [ $RELEASE_ALL ]; then
        BUILD_FLAGS="${BUILD_FLAGS} --build-ai-oneapi"
    fi
    sudo bash ./build-docker.sh  --image-tag $IMAGE_TAG --region ${CLOUDTIK_REGION} --python-version ${PYTHON_VERSION} \
        ${BUILD_FLAGS}
fi

# Default push
if [ ! $NO_PUSH ]; then
    for registry_region in ${registry_regions[@]};
    do
        DOCKER_REGISTRY=""
        if [ "${registry_region}" == "PRC" ]; then
            DOCKER_REGISTRY=${DOCKER_REGISTRY_PRC}
        fi

        PUSH_IMAGE_NAMES=()
        if [ $RELEASE_CLOUDTIK ] || [ $RELEASE_ALL ]; then
            PUSH_IMAGE_NAMES+=("cloudtik")
        fi
        if [ $RELEASE_SPARK ] || [ $RELEASE_ALL ]; then
            PUSH_IMAGE_NAMES+=("spark-runtime" "spark-runtime-benchmark")
        fi
        if [ $RELEASE_AI ] || [ $RELEASE_ALL ]; then
            PUSH_IMAGE_NAMES+=("spark-ai-runtime")
        fi
        if [ $RELEASE_AI_ONEAPI ] || [ $RELEASE_ALL ]; then
            PUSH_IMAGE_NAMES+=("spark-ai-oneapi")
        fi

        for image_name in ${PUSH_IMAGE_NAMES[@]}
        do
            image=${DOCKER_REGISTRY}cloudtik/$image_name:$IMAGE_TAG$GPU
            if [ "$(sudo docker images -q $image 2> /dev/null)" != "" ]; then
                sudo docker push $image
            fi
        done
    done
fi
