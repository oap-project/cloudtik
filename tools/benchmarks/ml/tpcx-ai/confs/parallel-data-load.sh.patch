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


usage() {
  echo "Usage: $0 <nodes> <copy_threads> <destination> [<file1/dir1> <file2/dir2> ...]"
  exit 1
}

if [ $# -eq 3 ]
then
  echo "No source file/directory specified (nothing to load). Skipping..."
  exit 0
elif [ $# -lt 3 ]
then
  usage
fi 

nodes=`cat $1 | grep -v '^ *#'`;
node_count=`echo $nodes | wc -w`

threads=$2
destination=$3
FILES_TO_LOAD=${@:4}

for src in $FILES_TO_LOAD; do  
  file=$(basename -- $src)
  for node in $nodes; do
    is_dir=`cloudtik head exec  --node-ip $node "if [[ -d $src ]]; then echo "1"; else echo "0"; fi" |  tail -n 2 |head -n 1 |tr -d "\r"` 
    if [[ $is_dir -eq 1 ]]
    then
      cloudtik head exec --node-ip $node "hdfs dfs -mkdir -p $destination/${file}.seq"
      cloudtik head exec --node-ip $node "${TPCx_AI_HOME_DIR}/tools/dir2seq.sh $src ${src}.seq && hdfs dfs -copyFromLocal -t $threads -f ${src}.seq $destination/${file}.seq/"'`hostname`' &
    else
      # a file with the same name but different contents is copied from the nodes
      # create a directory with the filename on hdfs and copy the files into that directory
      # to prevent overwriting the files are renamed to their resp. hostname
      # e.g. src/data/file.csv from host-a and host-b is copied to
      # dst/data/file.csv/host-a and dst/data/file.csv/host-b
      cloudtik head exec --node-ip $node "hdfs dfs -mkdir -p $destination/$file"
      cloudtik head exec --node-ip $node "hdfs dfs -copyFromLocal -t $threads -f $src $destination/$file/"'`hostname`' &
    fi
  done
done

# wait for data loading to finish on all nodes
echo "wait for data loading to finish"
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

