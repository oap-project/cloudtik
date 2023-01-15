#!/bin/bash

function set_head_address() {
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
}
