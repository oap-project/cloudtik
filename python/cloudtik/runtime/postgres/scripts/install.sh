#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

export PG_MAJOR=15

export USER_HOME=/home/$(whoami)
export RUNTIME_PATH=$USER_HOME/runtime
export POSTGRES_HOME=$RUNTIME_PATH/postgres

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function install_postgres() {
    if ! command -v postgres &> /dev/null
    then
        # make the "en_US.UTF-8" locale so postgres will be utf-8 enabled by default
        sudo apt-get -qq update -y >/dev/null \
          && sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
              --no-install-recommends locales libnss-wrapper zstd >/dev/null \
          && sudo rm -rf /var/lib/apt/lists/* \
	        && sudo localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
          && echo "export LANG=en_US.utf8" >> ${USER_HOME}/.bashrc

        # download the signing key
        # wget -O - -q https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
        # echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" \
        #     | sudo tee /etc/apt/sources.list.d/postgres.list

        set -e; \
        key='B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8'; \
        export GNUPGHOME="$(mktemp -d)"; \
        sudo gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key" >/dev/null 2>&1; \
        sudo mkdir -p /usr/local/share/keyrings/; \
        sudo gpg --batch --export --armor "$key" | sudo tee /usr/local/share/keyrings/postgres.gpg.asc >/dev/null; \
        sudo gpgconf --kill all; \
        rm -rf "$GNUPGHOME"
        echo "deb [ signed-by=/usr/local/share/keyrings/postgres.gpg.asc ] http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" \
            | sudo tee /etc/apt/sources.list.d/postgres.list >/dev/null

	      # install
        sudo apt-get -qq update -y >/dev/null \
          && sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
            --no-install-recommends postgresql-common >/dev/null \
          && sudo sed -ri 's/#(create_main_cluster) .*$/\1 = false/' /etc/postgresql-common/createcluster.conf \
          && sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
              --no-install-recommends "postgresql-$PG_MAJOR" "postgresql-client-$PG_MAJOR" >/dev/null \
          && sudo rm -f /etc/apt/sources.list.d/postgres.list \
          && echo "export PATH=/usr/lib/postgresql/$PG_MAJOR/bin:\$PATH" >> ${USER_HOME}/.bashrc
    fi
}

set_head_option "$@"
install_postgres
clean_install_cache
