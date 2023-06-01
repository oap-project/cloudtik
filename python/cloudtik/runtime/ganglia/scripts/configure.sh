#!/bin/bash

# Current bin directory
BIN_DIR=`dirname "$0"`
ROOT_DIR="$(dirname "$(dirname "$BIN_DIR")")"

args=$(getopt -a -o h:: -l head:: -- "$@")
eval set -- "${args}"

USER_HOME=/home/$(whoami)

# Util functions
. "$ROOT_DIR"/common/scripts/util-functions.sh

function prepare_base_conf() {
    source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
    output_dir=/tmp/ganglia/conf
    rm -rf  $output_dir
    mkdir -p $output_dir
    cp -r $source_dir/* $output_dir
}

function configure_diskstat() {
    node_disks=$(echo $(sudo hwinfo --disk --short | grep dev | awk '{print $1}' | cut -d "/" -f 3))
    sed -i "s/{cloudtik-node-disks}/${node_disks}/g" $output_dir/conf.d/diskstat.pyconf
    sudo cp $output_dir/conf.d/diskstat.pyconf /etc/ganglia/conf.d/diskstat.pyconf
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
        if (! grep -Fq 'gmond.head.conf' /etc/init.d/ganglia-monitor)
        then
            sudo sed -i '/\NAME.pid/ a \ \ start-stop-daemon --start --quiet --startas $DAEMON --name $NAME.head -- --conf /etc/ganglia/gmond.head.conf --pid-file /var/run/$NAME.head.pid' /etc/init.d/ganglia-monitor
        fi

        # Configure ganlia monitor python for worker
        configure_diskstat
    else
        # Configure ganglia monitor for worker
        sed -i "s/{cloudtik-cluster-name}/${cluster_name_worker}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-head-address}/${HEAD_ADDRESS}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-bind-address}/${NODE_IP_ADDRESS}/g" $output_dir/gmond.node.conf
        sed -i "s/{cloudtik-port}/8649/g" $output_dir/gmond.node.conf
        sudo cp $output_dir/gmond.node.conf /etc/ganglia/gmond.conf

        # Configure ganlia monitor python for worker
        configure_diskstat
    fi
}

set_head_option "$@"
set_head_address
set_node_ip_address
configure_ganglia

exit 0
