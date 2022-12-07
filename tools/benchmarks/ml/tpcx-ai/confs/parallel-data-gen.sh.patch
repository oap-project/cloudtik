#!/bin/bash

#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


DEFAULT_PDGF_PATH="`pwd`/lib/pdgf"

usage() {
  echo "Usage: $0 [-h <nodes file>] [-o <pdgf options>]"
  exit 1
}

while getopts "h:o:p:" arg
do
  case "${arg}" in
    h)
      nodes_file="$OPTARG"
      ;;
    p)
      pdgf="$OPTARG"
      ;;
    o)
      pdgf_options="$OPTARG"
      ;;
  esac
done

shift $((OPTIND-1))

if [ -z "$nodes_file" ]; then
  usage
fi

if [ -n "$pdgf" ]; then
  TPCXAI_PDGF_PATH="$pdgf"
elif [ -n "$PDGF_PATH" ]; then
  TPCXAI_PDGF_PATH=$PDGF_PATH
else
  TPCXAI_PDGF_PATH=$DEFAULT_PDGF_PATH
fi

nodes=`cat $nodes_file | grep -v '^ *#'`;
node_count=`echo $nodes | wc -w`

# kill all subproccesses when exit
trap 'jobs -p | xargs -r kill' EXIT;

# check the hosts
echo "Check environment"
## check if host is reachable with passwordless ssh
failed_hosts=()
for node in $nodes
do
  #ssh -oNumberOfPasswordPrompts=0 "$node" "echo check connection" > /dev/null
  cloudtik head exec --node-ip $node  "echo check connection" > /dev/null
  if [ $? -ne 0 ]; then
    failed_hosts+=( "$node" )
  fi
done
if [ ${#failed_hosts[@]} -ne 0 ]; then
  echo "Failed to connect to: ${failed_hosts[@]}"
  exit 1
else
  echo "All hosts are reachable"
fi

## check for binaries (java, pdgf, hdfs)
BINARIES="java"
failed_hosts=()
for node in $nodes
do
  for bin in $BINARIES
  do
    #ssh $node -q "hash $bin" 2> /dev/null
    cloudtik head exec --node-ip $node "hash $bin" 
    if [ "$?" -ne 0 ]; then
      echo "$node: command not found: $bin"
    fi
  done
done

# initialize nodes
## accept PDGF EULA on all nodes
for node in $nodes
do
  cloudtik head exec --node-ip $node "echo \"# auto-accepted EULA by $0 on `date`\" >> $TPCXAI_PDGF_PATH/Constants.properties"
  cloudtik head exec --node-ip $node "echo \"IS_EULA_ACCEPTED=true\" >> $TPCXAI_PDGF_PATH/Constants.properties"
done

node_number=1
for node in $nodes
do
  echo "$node_number/$node_count: $node";
  pdgf_options_esc=\'${pdgf_options//\'/\'\\\'\'}\'
  echo "pdgf options: $pdgf_options"
  cloudtik head exec --node-ip $node "cd $TPCXAI_PDGF_PATH; java -Djava.awt.headless=true $TPCxAI_PDGF_JAVA_OPTS -jar $TPCXAI_PDGF_PATH/pdgf.jar -nn $node_number -nc $node_count -ns $pdgf_options" > >(sed -e "s/^/$node: /;") 2> >(sed -e 's/^/ ERR: /;') & let node_number=node_number+1;
done

# wait for data generation to finish on all nodes
echo "wait for data generation to finish"
FAIL=0
for job in `jobs -p`
do
  wait $job || let FAIL+=1
done

if [ $FAIL -eq 0 ]; then
  echo "DONE"
elif [ $FAIL -eq 1 ]; then
  echo "one job failed"
else
  echo "$FAIL jobs failed"
fi



