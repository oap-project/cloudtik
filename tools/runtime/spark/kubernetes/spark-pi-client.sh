#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
WORK_DIR="$(dirname "$0")"

PARTITIONS=100

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -p|--partitions)
    shift 1 # past argument
    PARTITIONS=$1
    shift 1 # past value
    ;;
    *)    # completed this shell arguments processing
    break
    ;;
esac
done

bash ${WORK_DIR}/spark-submit-client.sh \
  "$@" \
  --name spark-pi \
  --class org.apache.spark.examples.SparkPi \
  local:///opt/runtime/spark/examples/jars/spark-examples.jar ${PARTITIONS}
