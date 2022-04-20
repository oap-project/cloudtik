#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "${OSTYPE}" = msys ]; then
    echo "WARNING: ${0##*/} is not recommended on MSYS2, as MSYS2 alters the build environment."
fi

# and copy the binary to python/cloudtik/core/thirdparty/redis/redis-server
BUILD_DIR=${SCRIPT_DIR}/build
THIRDPARTY_DIR=${BUILD_DIR}/thirdparty
mkdir -p ${THIRDPARTY_DIR}

# download redis source code
rm -f ${THIRDPARTY_DIR}/redis-stable.tar.gz
wget -O ${THIRDPARTY_DIR}/redis-stable.tar.gz http://download.redis.io/redis-stable.tar.gz
tar xvzf ${THIRDPARTY_DIR}/redis-stable.tar.gz  -C ${THIRDPARTY_DIR}

# compile redis
REDIS_BUILD_DIR=${THIRDPARTY_DIR}/redis-stable
(cd ${REDIS_BUILD_DIR} && make)

# copy executable to target folder
mkdir -p ${SCRIPT_DIR}/python/cloudtik/core/thirdparty/redis
mv ${REDIS_BUILD_DIR}/src/redis-server  ${SCRIPT_DIR}/python/cloudtik/core/thirdparty/redis/redis-server
chmod +x ${SCRIPT_DIR}/python/cloudtik/core/thirdparty/redis/redis-server

# clear up the build directory
rm -r -f -- ${REDIS_BUILD_DIR}

mkdir -p ${SCRIPT_DIR}/python/cloudtik/runtime/spark/conf
cp -r ${SCRIPT_DIR}/runtime/spark/conf ${SCRIPT_DIR}/python/cloudtik/runtime/spark/
cp -r ${SCRIPT_DIR}/runtime/spark/scripts ${SCRIPT_DIR}/python/cloudtik/runtime/spark/

# build pip wheel
cd ${SCRIPT_DIR}/python

# Update the commit sha
if [ -n "${TRAVIS_COMMIT}" ]; then
    CLOUDTIK_COMMIT_SHA=$TRAVIS_COMMIT
fi

if [ ! -n "$CLOUDTIK_COMMIT_SHA" ]; then
    CLOUDTIK_COMMIT_SHA=$(which git >/dev/null && git rev-parse HEAD)
fi

if [ ! -z "$CLOUDTIK_COMMIT_SHA" ]; then
    sed -i.bak "s/__commit__ = \".*\"/__commit__ = \"$CLOUDTIK_COMMIT_SHA\"/g" ./cloudtik/__init__.py && rm ./cloudtik/__init__.py.bak
fi

bash ./build-wheel-manylinux2014.sh
