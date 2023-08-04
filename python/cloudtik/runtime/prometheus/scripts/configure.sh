#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
PROMETHEUS_HOME=$RUNTIME_PATH/prometheus

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(dirname "${BIN_DIR}")/conf
    output_dir=/tmp/prometheus/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function check_prometheus_installed() {
    if [ ! -f "${PROMETHEUS_HOME}/prometheus" ]; then
        echo "Prometheus is not installed for prometheus command is not available."
        exit 1
    fi
}

function update_local_file() {
    cp -r $output_dir/scrape-config-local-file.yaml ${PROMETHEUS_CONFIG_DIR}/scrape-config-local-file.yaml
}

function update_local_consul() {
  cp -r $output_dir/scrape-config-local-consul.yaml ${PROMETHEUS_CONFIG_DIR}/scrape-config-local-consul.yaml
}

function update_workspace_consul() {
  cp -r $output_dir/scrape-config-workspace-consul.yaml ${PROMETHEUS_CONFIG_DIR}/scrape-config-workspace-consul.yaml
}

function update_federation_consul() {
  # Federation will also scrape local cluster
  update_local_consul
  cp -r $output_dir/scrape-config-federation-consul.yaml ${PROMETHEUS_CONFIG_DIR}/scrape-config-federation-consul.yaml
}

function configure_prometheus() {
    prepare_base_conf
    prometheus_output_dir=$output_dir
    config_template_file=${output_dir}/prometheus.yaml

    mkdir -p ${PROMETHEUS_HOME}/logs

    PROMETHEUS_CONFIG_DIR=${PROMETHEUS_HOME}/conf
    mkdir -p ${PROMETHEUS_CONFIG_DIR}

    sed -i "s#{%prometheus.home%}#${PROMETHEUS_HOME}#g" `grep "{%prometheus.home%}" -rl ${output_dir}`
    sed -i "s#{%workspace.name%}#${CLOUDTIK_WORKSPACE}#g" `grep "{%workspace.name%}" -rl ${output_dir}`
    sed -i "s#{%cluster.name%}#${CLOUDTIK_CLUSTER}#g" `grep "{%cluster.name%}" -rl ${output_dir}`

    if [ "${PROMETHEUS_SCRAPE_SCOPE}" == "workspace" ]; then
        if [ "${PROMETHEUS_SERVICE_DISCOVERY}" == "CONSUL" ]; then
            update_workspace_consul
        fi
    elif [ "${PROMETHEUS_SCRAPE_SCOPE}" == "federation" ]; then
        if [ "${PROMETHEUS_SERVICE_DISCOVERY}" == "CONSUL" ]; then
            update_federation_consul
        fi
    else
        # local scope
        if [ "${PROMETHEUS_SERVICE_DISCOVERY}" == "CONSUL" ]; then
            update_local_consul
        elif [ "${PROMETHEUS_SERVICE_DISCOVERY}" == "FILE" ]; then
            update_local_file
        fi
    fi

    cp -r $output_dir/prometheus.yaml ${PROMETHEUS_CONFIG_DIR}/prometheus.yaml
}

set_head_option "$@"
check_prometheus_installed
set_head_address
set_node_ip_address
configure_prometheus

exit 0
