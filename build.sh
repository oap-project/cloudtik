#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "${OSTYPE}" = msys ]; then
    echo "WARNING: ${0##*/} is not recommended on MSYS2, as MSYS2 alters the build environment."
fi

# and copy the binary to python/cloudtik/core/thirdparty/redis/redis-server
BUILD_DIR=${SCRIPT_DIR}/build
THIRDPARTY_DIR=${BUILD_DIR}/thirdparty
mkdir -p ${THIRDPARTY_DIR}

function compile_redis_server() {
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
}

function compile_ganglia_for_python() {
    # download ganglia-monitor-core
    rm -rf /tmp/monitor-core && git clone https://github.com/ganglia/monitor-core.git /tmp/monitor-core &&  cd /tmp/monitor-core && git checkout release/3.6

    # install prerequisit before compiling
    sudo -E apt-get -yq --no-install-suggests --no-install-recommends install \
        libapr1-dev libaprutil1-dev libconfuse-dev libexpat1-dev libpcre3-dev libssl-dev librrd-dev libperl-dev libtool m4 gperf zlib1g-dev pkg-config libtool python2.7-dev automake make

    # compile ganglia-monitor-core
    ./bootstrap && ./configure --with-gmetad --enable-status --with-python=/usr/bin/python2.7 && make

    # upload binary to remote repository
    # curl -X PUT --upload-file ./gmond/modules/python/.libs/modpython.so  https://d30257nes7d4fq.cloudfront.net/downloads/cloudtik/ganglia/modpython.so
}

compile_redis_server
compile_ganglia_for_python

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
