#!/bin/bash

function set_head_address() {
    if [ -z "${HEAD_ADDRESS}" ]; then
        if [ $IS_HEAD_NODE == "true" ]; then
            if [ ! -n "${CLOUDTIK_NODE_IP}" ]; then
                HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
            else
                HEAD_ADDRESS=${CLOUDTIK_NODE_IP}
            fi
        else
            if [ ! -n "${CLOUDTIK_HEAD_IP}" ]; then
                echo "Error: CLOUDTIK_HEAD_IP environment variable should be set."
                exit 1
            else
                HEAD_ADDRESS=${CLOUDTIK_HEAD_IP}
            fi
        fi
    fi
}

function set_node_ip_address() {
    if [ -z "${NODE_IP_ADDRESS}" ]; then
        if [ ! -n "${CLOUDTIK_NODE_IP}" ]; then
            NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
        else
            NODE_IP_ADDRESS=${CLOUDTIK_NODE_IP}
        fi
    fi
}

function clean_install_cache() {
    (sudo rm -rf /var/lib/apt/lists/* \
        && sudo apt-get clean \
        && which conda > /dev/null && conda clean -itqy)
}
