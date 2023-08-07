#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

# import util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

if ! command -v consul &> /dev/null
then
    echo "Consul is not installed for consul command is not available."
    exit 1
fi

set_head_option "$@"
set_service_command "$@"

USER_HOME=/home/$(whoami)
RUNTIME_PATH=$USER_HOME/runtime
CONSUL_HOME=$RUNTIME_PATH/consul
CONSUL_PID_FILE=${CONSUL_HOME}/consul-agent.pid

case "$SERVICE_COMMAND" in
start)
    CONSUL_CONFIG_DIR=${CONSUL_HOME}/consul.d
    CONSUL_LOG_FILE=${CONSUL_HOME}/logs/consul-agent.log
    # Run server or client agent on each node
    nohup consul agent \
        -config-dir=${CONSUL_CONFIG_DIR} \
        -log-file=${CONSUL_LOG_FILE} \
        -pid-file=${CONSUL_PID_FILE} >/dev/null 2>&1 &
    ;;
stop)
    # Stop server or client agent
    if [ -f "${CONSUL_PID_FILE}" ]; then
        AGENT_PID=$(cat ${CONSUL_PID_FILE})
        if [ -n "${AGENT_PID}" ]; then
          echo "Stopping Consul agent..."
          # SIGTERM = 15
          kill -15 ${AGENT_PID} >/dev/null 2>&1
        fi
    fi
    ;;
-h|--help)
    echo "Usage: $0 start|stop --head" >&2
    ;;
*)
    echo "Usage: $0 start|stop --head" >&2
    ;;
esac

exit 0
