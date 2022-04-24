#!/bin/bash

args=$(getopt -a -o h::p: -l head::,head_address:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false
USER_HOME=/home/$(whoami)

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
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
    if [ ! -n "${HEAD_ADDRESS}" ]; then
        HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
    fi
}

function configure_ganglia() {
    cluster_name_head="CloudTik-Head"
    cluster_name="CloudTik-Workers"
    if [ $IS_HEAD_NODE == "true" ]; then
        # configure ganglia gmetad
        sudo sed -i "0,/# default: There is no default value/s//data_source \"${cluster_name_head}\" ${HEAD_ADDRESS}:8650/" /etc/ganglia/gmetad.conf
        sudo sed -i "s/data_source \"my cluster\" localhost/data_source \"${cluster_name}\" ${HEAD_ADDRESS}/g" /etc/ganglia/gmetad.conf
        sudo sed -i "s/# gridname \"MyGrid\"/gridname \"CloudTik\"/g" /etc/ganglia/gmetad.conf

        # Configure ganglia monitor
        sudo sed -i "s/send_metadata_interval = 0/send_metadata_interval = 30/g" /etc/ganglia/gmond.conf
        # replace the first occurrence of "mcast_join = 239.2.11.71" with "host = HEAD_IP"
        sudo sed -i "0,/mcast_join = 239.2.11.71/s//host = ${HEAD_ADDRESS}/" /etc/ganglia/gmond.conf
        # comment out the second occurrence
        sudo sed -i "s/mcast_join = 239.2.11.71/\/*mcast_join = 239.2.11.71*\//g" /etc/ganglia/gmond.conf
        sudo sed -i "s/bind = 239.2.11.71/bind = ${HEAD_ADDRESS}/g" /etc/ganglia/gmond.conf
        sudo sed -i "/tcp_accept_channel {/ a \ \ bind = ${HEAD_ADDRESS}" /etc/ganglia/gmond.conf

        # Make a copy for head cluster after common modifications
        sudo cp /etc/ganglia/gmond.conf /etc/ganglia/gmond.head.conf

        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name}\"/g" /etc/ganglia/gmond.conf
        # Disable udp_send_channel
        sudo sed -i "s/^udp_send_channel {/\/*udp_send_channel {/g" /etc/ganglia/gmond.conf
        sudo sed -i "s/^\/\* You can specify as many udp_recv_channels/\*\/\/\* You can specify as many udp_recv_channels/g" /etc/ganglia/gmond.conf

        # Modifications for head cluster
        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name_head}\"/g" /etc/ganglia/gmond.head.conf
        sudo sed -i "s/port = 8649/port = 8650/g" /etc/ganglia/gmond.head.conf

        # Configure apache2 for ganglia
        sudo cp /etc/ganglia-webfrontend/apache.conf /etc/apache2/sites-enabled/ganglia.conf
        # Fix the ganglia bug: https://github.com/ganglia/ganglia-web/issues/324
        # mention here: https://bugs.launchpad.net/ubuntu/+source/ganglia-web/+bug/1822048
        sudo sed -i "s/\$context_metrics = \"\";/\$context_metrics = array();/g" /usr/share/ganglia-webfrontend/cluster_view.php

        # Add gmond start command for head in service
        sudo sed -i '/\NAME.pid/ a start-stop-daemon --start --quiet --startas $DAEMON --name $NAME.head -- --conf /etc/ganglia/gmond.head.conf --pid-file /var/run/$NAME.head.pid' /etc/init.d/ganglia-monitor
    else
        # Configure ganglia monitor
        sudo sed -i "s/send_metadata_interval = 0/send_metadata_interval = 30/g" /etc/ganglia/gmond.conf
        sudo sed -i "s/name = \"unspecified\"/name = \"${cluster_name}\"/g" /etc/ganglia/gmond.conf
        # replace the first occurrence of "mcast_join = 239.2.11.71" with "host = HEAD_IP"
        sudo sed -i "0,/mcast_join = 239.2.11.71/s//host = ${HEAD_ADDRESS}/" /etc/ganglia/gmond.conf
    fi
}


set_head_address
configure_ganglia
