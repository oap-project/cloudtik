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

if [ ! -n "${SPARK_HOME}" ]; then
  echo "SPARK_HOME environment variable is not set."
  exit 1
fi

SPARK_CONF=${WORK_DIR}/conf

K8S_API_SERVER_ENDPOINT=localhost:8443
K8S_NAMESPACE=default
SPARK_DRIVER_SERVICE_ACCOUNT=cloudtik-head-service-account
SPARK_EXECUTOR_SERVICE_ACCOUNT=cloudtik-worker-service-account
CONTAINER_IMAGE=cloudtik/spark-kubernetes:nightly

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -a|--api-server)
    shift 1 # past argument
    K8S_API_SERVER_ENDPOINT=$1
    shift 1 # past value
    ;;
    -n|--namespace)
    shift 1 # past argument
    K8S_NAMESPACE=$1
    shift 1 # past value
    ;;
    -d|--driver-service-account)
    shift 1 # past argument
    SPARK_DRIVER_SERVICE_ACCOUNT=$1
    shift 1 # past value
    ;;
    -e|--executor-service-account)
    shift 1 # past argument
    SPARK_EXECUTOR_SERVICE_ACCOUNT=$1
    shift 1 # past value
    ;;
    -i|--image)
    shift 1 # past argument
    CONTAINER_IMAGE=$1
    shift 1 # past value
    ;;
    -s|--spark-conf)
    shift 1 # past argument
    SPARK_CONF=$1
    shift 1 # past value
    ;;
    -h|--help)
    shift 1 # past argument
    echo "Usage: spark-submit.sh --api-server k8s-api-server-endpoint --namespace k8s-namespace --driver-service-account spark-driver-service-account --executor-service-account spark-executor-service-account --image image:tag --spark-conf spark-conf-dir --help other spark configurations"
    exit 1
    ;;
    *)    # completed this shell arguments processing
    break
    ;;
esac
done

echo "Use Spark configuration at ${SPARK_CONF}"
export SPARK_CONF_DIR=${SPARK_CONF}

$SPARK_HOME/bin/spark-submit \
    --master k8s://$K8S_API_SERVER_ENDPOINT \
    --deploy-mode cluster \
    --conf spark.kubernetes.namespace=$K8S_NAMESPACE \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=$SPARK_DRIVER_SERVICE_ACCOUNT \
    --conf spark.kubernetes.authenticate.executor.serviceAccountName=$SPARK_EXECUTOR_SERVICE_ACCOUNT \
    --conf spark.kubernetes.container.image=$CONTAINER_IMAGE \
    "$@"
