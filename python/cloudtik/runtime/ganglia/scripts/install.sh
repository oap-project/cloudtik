#!/bin/bash

args=$(getopt -a -o h::p: -l head:: -- "$@")
eval set -- "${args}"

IS_HEAD_NODE=false

while true
do
    case "$1" in
    --head)
        IS_HEAD_NODE=true
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done


function install_ganglia_monitor_python() {
    wget -q --show-progress -P /tmp https://d30257nes7d4fq.cloudfront.net/downloads/ganglia/modpython.so
    sudo cp /tmp/modpython.so /usr/lib/ganglia/ && sudo chmod 644 /usr/lib/ganglia/modpython.so
    sudo apt-get install -y  ganglia-monitor-python python2.7-dev > /dev/null
}

function install_ganglia_server() {
    # Simply do the install, if they are already installed, it doesn't take time
    sudo apt-get -qq update -y > /dev/null
    sudo apt-get -qq install -y apache2 php libapache2-mod-php php-common php-mbstring php-gmp php-curl php-intl php-xmlrpc php-zip php-gd php-mysql php-xml > /dev/null
    sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ganglia-monitor rrdtool gmetad ganglia-webfrontend > /dev/null
    install_ganglia_monitor_python
}

function install_ganglia_client() {
    sudo apt-get -qq update -y > /dev/null
    sudo apt-get -qq install -y ganglia-monitor > /dev/null
    install_ganglia_monitor_python
}

function install_ganglia() {
    if [ $IS_HEAD_NODE == "true" ];then
        install_ganglia_server
    else
        install_ganglia_client
    fi
}

install_ganglia
