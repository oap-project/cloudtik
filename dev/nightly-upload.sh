#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

# Import the default vars
. "$SCRIPT_DIR"/set-default-vars.sh

cd $CLOUDTIK_HOME

CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ./python/cloudtik/__init__.py)
PYTHON_TAG=${PYTHON_VERSION//./}

arch=$(uname -m)

CLOUDTIK_NIGHTLY_WHEEL=cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_${arch}.nightly.whl
aws s3 cp ./python/dist/$CLOUDTIK_NIGHTLY_WHEEL s3://cloudtik/downloads/cloudtik/

aws cloudfront create-invalidation --distribution-id E3703BSG9BICN1 --paths "/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp${PYTHON_TAG}-cp${PYTHON_TAG}-manylinux2014_${arch}.nightly.whl"
