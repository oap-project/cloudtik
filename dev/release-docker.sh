#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

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
    *)
        echo "Usage: build-docker.sh [ --image-tag ]"
        exit 1
    esac
    shift
done


cd $CLOUDTIK_HOME
source /home/ubuntu/anaconda3/bin/activate cloudtik_py37

sudo bash ./build-docker.sh --image-tag $IMAGE_TAG --build-spark --build-ml --build-spark-benchmark

sudo docker push cloudtik/cloudtik:$IMAGE_TAG
sudo docker push cloudtik/spark-runtime:$IMAGE_TAG
sudo docker push cloudtik/spark-ml-runtime:$IMAGE_TAG
sudo docker push cloudtik/spark-runtime-benchmark:$IMAGE_TAG
