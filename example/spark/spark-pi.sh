#!/bin/bash

function show_usage() {
    echo "Usage: spark-pi.sh cluster-config-file [--slices number-of-slices] [--help]"
}

pi_slices=100

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
    -s|--slices)
        shift 1 # past argument
        pi_slices=$1
        shift 1 # past value
        ;;
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
    "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi \$SPARK_HOME/examples/jars/spark-examples.jar ${pi_slices}"
