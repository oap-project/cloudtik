#!/bin/bash

args=$(getopt -a -o h::p: -l head::,node_ip_address::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --node_ip_address)
        NODE_IP_ADDRESS=$2
        shift
        ;;
    -h|--head_address)
        HEAD_ADDRESS=$2
        shift
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done


function set_head_address() {
    if [ $IS_HEAD_NODE == "true" ]; then
        if [ ! -n "${NODE_IP_ADDRESS}" ]; then
            HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
        else
            HEAD_ADDRESS=${NODE_IP_ADDRESS}
        fi
    else
        if [ ! -n "${HEAD_ADDRESS}" ]; then
            # Error: no head address passed
            echo "Error: head ip address should be passed."
            exit 1
        fi
    fi
}

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/ganglia/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function configure_ganglia() {
    prepare_base_conf

    cluster_name_worker="CloudTik-Workers"
    if [ $IS_HEAD_NODE == "true" ]; then
        cloudtik_grid_name="CloudTik"
        cluster_name_head="CloudTik-Head"
        head_data_source_address=${HEAD_ADDRESS}:8650
        worker_data_source_address=${HEAD_ADDRESS}:8649

        # configure ganglia gmetad
        sed -i "s/{cloudtik-grid-name}/${cloudtik_grid_name}/g" $output_dir/gmetad.conf
        sed -i "s/{cloudtik-head-data-source-name}/${cluster_name_head}/g" $output_dir/gmetad.conf
        sed -i "s/{cloudtik-worker-data-source-name}/${cluster_name_worker}/g" $output_dir/gmetad.conf
        sed -i "s/{cloudtik-head-data-source-address}/${head_data_source_address}/g" $output_dir/gmetad.conf
        sed -i "s/{cloudtik-worker-data-source-address}/${worker_data_source_address}/g" $output_dir/gmetad.conf
        sudo cp $output_dir/gmetad.conf /etc/ganglia/gmetad.conf

        # configure gmond for head cluster
        cp $output_dir/gmond.node.conf $output_dir/gmond.head.conf
        sed -i "s/{cloudtik-cluster-name}/${cluster_name_head}/g" $output_dir/gmond.head.conf
        sed -i "s/{cloudtik-head-address}/${HEAD_ADDRESS}/g" $output_dir/gmond.head.conf
        sed -i "s/{cloudtik-bind-address}/${NODE_IP_ADDRESS}/g" $output_dir/gmond.head.conf
        sed -i "s/{cloudtik-port}/8650/g" $output_dir/gmond.head.conf
        sudo cp $output_dir/gmond.head.conf /etc/ganglia/gmond.head.conf

        # configure gmond for worker cluster
        sed -i "s/{cloudtik-cluster-name}/${cluster_name_worker}/g" $output_dir/gmond.conf
        sed -i "s/{cloudtik-bind-address}/${NODE_IP_ADDRESS}/g" $output_dir/gmond.conf
        sed -i "s/{cloudtik-port}/8649/g" $output_dir/gmond.conf
        sudo cp $output_dir/gmond.conf /etc/ganglia/gmond.conf

        # Configure apache2 for ganglia
        sudo cp /etc/ganglia-webfrontend/apache.conf /etc/apache2/sites-enabled/ganglia.conf
        # Fix the ganglia bug: https://github.com/ganglia/ganglia-web/issues/324
        # mention here: https://bugs.launchpad.net/ubuntu/+source/ganglia-web/+bug/1822048
        sudo sed -i "s/\$context_metrics = \"\";/\$context_metrics = array();/g" /usr/share/ganglia-webfrontend/cluster_view.php

        # Add gmond start command for head in service
        sudo sed -i '/\NAME.pid/ a start-stop-daemon --start --quiet --startas $DAEMON --name $NAME.head -- --conf /etc/ganglia/gmond.head.conf --pid-file /var/run/$NAME.head.pid' /etc/init.d/ganglia-monitor
    else
        # Configure ganglia monitor for woker
        sed -i "s/{cloudtik-cluster-name}/${cluster_name_worker}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-head-address}/${HEAD_ADDRESS}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-bind-address}/${NODE_IP_ADDRESS}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-port}/8649/g" $output_dir/gmond.node.conf
        sudo cp $output_dir/gmond.node.conf /etc/ganglia/gmond.conf
    fi
}

set_head_address
configure_ganglia
