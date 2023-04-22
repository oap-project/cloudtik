#!/bin/bash
ZOOKEEPER_CLI=$ZOOKEEPER_HOME/bin/zkCli.sh

# get one of the zookeeper server
server_ip=$(cloudtik head worker-ips --separator ' ' | awk '{print $1;}')

if [ $server_ip == "" ]; then
    echo "No zookeeper server started."
    exit 1
fi

# execute create command
$ZOOKEEPER_CLI -server $server_ip:2181 create /abc

# execute list to make sure it created
$ZOOKEEPER_CLI -server $server_ip:2181 ls /
