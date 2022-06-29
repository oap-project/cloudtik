#!/bin/bash

args=$(getopt -a -o a:c:hy -l action:,config:,help,yes -- "$@")
eval set -- "${args}"

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

function check_cloudtik_environment() {
    which cloudtik > /dev/null || (echo "Cloudtik is not found. Please install cloudtik first!"; exit 1)
}

function check_aws_resource_action() {
    AWS_RESOURCE_ALLOW_ACTIONS=( create-workspace delete-workspace start-cluster stop-cluster )
    if [ $(contains "${AWS_RESOURCE_ALLOW_ACTIONS[@]}" "$ACTION") == "y" ]; then
        echo "Action $ACTION is allowed for aws resource."
    else
        echo "Action $ACTION is not allowed for aws resource. Supported action: ${AWS_RESOURCE_ALLOW_ACTIONS[*]}."
        exit 1
    fi
}

function check_aws_resource_config() {
    if [ -f "${CONFIG}" ]; then
         echo "The config file exist"
    else
         echo "The config file doesn't exist"
	 exit 1
    fi
}

function create_aws_workspace() {
    cloudtik workspace create $CONFIG $CONFIRM
}

function delete_aws_workspace() {
    cloudtik workspace delete $CONFIG $CONFIRM
}

function start_aws_cluster() {
    cloudtik start $CONFIG $CONFIRM
}

function stop_aws_cluster() {
    cloudtik stop $CONFIG $CONFIRM
}

function usage() {
    echo "Usage: $0 -a|--action [create-workspace|delete-workspace|start-cluster|stop-cluster] -c|--config [your.yaml] -y|--yes" >&2
    echo "Usage: $0 -h|--help"
}


while true
do
    case "$1" in
    -a|--action)
        ACTION=$2
        shift
        ;;
    -c|--config)
        CONFIG=$2
        shift
        ;;
    -y|--yes)
        CONFIRM="-y"
        ;;
    -h|--help)
        shift
        usage
        exit 0
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

check_cloudtik_environment
check_aws_resource_action
check_aws_resource_config

if [ "${ACTION}" == "create-workspace" ];then
    create_aws_workspace
elif [ "${ACTION}" == "delete-workspace" ];then
    delete_aws_workspace
elif [ "${ACTION}" == "start-cluster" ];then
    start_aws_cluster
elif [ "${ACTION}" == "stop-cluster" ];then
    stop_aws_cluster
else
    usage
    exit 1
fi
