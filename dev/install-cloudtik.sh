#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CLOUDTIK_HOME=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )
CONDA_HOME=$( cd -- "$( dirname -- "$( dirname -- "$(which conda)" )" )" &> /dev/null && pwd )

CLOUDTIK_VERSION=""

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    --version)
        # Override for the version.
        shift
        CLOUDTIK_VERSION=$1
        ;;
    --nightly)
        FROM_NIGHTLY=YES
        ;;
    --local)
        FROM_LOCAL=YES
        ;;
    --cloud)
        FROM_CLOUD=YES
        ;;
    --reinstall)
        REINSTALL=YES
        ;;
    *)
        echo "Usage: install.sh [ --version version ] [ --nightly ] [ --local ] [ --cloud ] [ --reinstall ]"
        exit 1
    esac
    shift
done


if [ $REINSTALL ]; then
    #conda activate cloudtik
    source $CONDA_HOME/bin/activate cloudtik
    pip uninstall cloudtik -y
else
    # activate base and delete 
    #conda activate
    source $CONDA_HOME/bin/activate base
    conda env remove -n cloudtik
    
    # creae env and install cloudtik
    conda create -n cloudtik -y python=3.7
    #conda activate cloudtik
    source $CONDA_HOME/bin/activate cloudtik
fi

arch=$(uname -m)

if [ $FROM_NIGHTLY ] || [ $FROM_LOCAL ] || [ $FROM_CLOUD ]; then
    if [ "$CLOUDTIK_VERSION" == "" ]; then
        CLOUDTIK_VERSION=$(sed -n 's/__version__ = \"\(..*\)\"/\1/p' ${CLOUDTIK_HOME}/python/cloudtik/__init__.py)
    fi
fi

if [ $FROM_NIGHTLY ]; then
    if [ $FROM_LOCAL ]; then
        pip install -U "cloudtik[all] @ file://${CLOUDTIK_HOME}/python/dist/cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.nightly.whl"
    else
        pip install -U "cloudtik[all] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.nightly.whl"
    fi
else
    if [ $FROM_LOCAL ]; then
        pip install -U "cloudtik[all] @ file://${CLOUDTIK_HOME}/python/dist/cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.whl"
    elif [ $FROM_CLOUD ]; then
        pip install -U "cloudtik[all] @ https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/cloudtik-${CLOUDTIK_VERSION}-cp37-cp37m-manylinux2014_${arch}.whl"
    else
    	if [ "$CLOUDTIK_VERSION" == "" ]; then
            pip install cloudtik[all]
    	else
            pip install cloudtik[all]==$CLOUDTIK_VERSION
    	fi
    fi
fi
