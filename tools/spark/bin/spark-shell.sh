#!/bin/bash

function show_usage() {
  echo "Usage: spark-shell.sh cluster-config-file [--help]"
}

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    shift 1 # past argument
    show_usage
    exit 1
    ;;
    *)    # cluster config file
    cluster_config_file=$1
    shift 1 # past argument
    ;;
esac
done

if [ -z "$cluster_config_file" ]
then
      echo "Error: cluster config file is not specified."
      show_usage
      exit 1
fi

cloudtik exec $cluster_config_file  \
  "spark-shell"
