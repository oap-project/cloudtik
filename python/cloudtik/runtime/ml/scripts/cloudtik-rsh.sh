#!/usr/bin/env bash
node_ip=$1
shift
cmd_str="$*"

cloudtik head exec "$cmd_str" --node-ip=$node_ip
