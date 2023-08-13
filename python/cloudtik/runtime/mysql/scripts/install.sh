#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export MYSQL_MAJOR=8.0
export MYSQL_VERSION=8.0.34-1debian11

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
export MYSQL_HOME=$RUNTIME_PATH/mysql

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_mysql() {
    if ! command -v mysqld &> /dev/null
    then
        # download the signing key
        key='859BE8D7C586F538430B19C2467B942D3A79BD29'; \
        export GNUPGHOME="$(mktemp -d)"; \
        sudo gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
        sudo mkdir -p /etc/apt/keyrings; \
        sudo gpg --batch --export "$key" | sudo tee /etc/apt/keyrings/mysql.gpg >/dev/null; \
        sudo gpgconf --kill all; \
        rm -rf "$GNUPGHOME"

	      # install
        echo "deb [ signed-by=/etc/apt/keyrings/mysql.gpg ] http://repo.mysql.com/apt/debian/ bullseye mysql-8.0" \
          | sudo tee /etc/apt/sources.list.d/mysql.list > /dev/null
        { \
          echo mysql-community-server mysql-community-server/data-dir select ''; \
          echo mysql-community-server mysql-community-server/root-pass password ''; \
          echo mysql-community-server mysql-community-server/re-root-pass password ''; \
          echo mysql-community-server mysql-community-server/remove-test-db select false; \
        } | sudo debconf-set-selections \
        && sudo apt-get update -y \
        && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
          mysql-community-client="${MYSQL_VERSION}" \
          mysql-community-server-core="${MYSQL_VERSION}" \
        && sudo rm -f /etc/apt/sources.list.d/mysql.list
    fi
}

set_head_option "$@"
install_mysql
clean_install_cache
