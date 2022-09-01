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

SPARK_CONF=${WORK_DIR}/conf/

K8S_NAMESPACE=default
SPARK_DRIVER_SERVICE_ACCOUNT=cloudtik-head-service-account
SPARK_EXECUTOR_SERVICE_ACCOUNT=cloudtik-worker-service-account
CONTAINER_IMAGE=cloudtik/spark-kubernetes:nightly

ACTION=shell

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
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
    -s|--spark_conf)
    shift 1 # past argument
    SPARK_CONF=$1
    shift 1 # past value
    ;;
    -h|--help)
    shift 1 # past argument
    echo "Usage: spark-[shell|sql|submit]-client.sh start|stop --namespace k8s-namespace --driver-service-account spark-driver-service-account --executor-service-account spark-executor-service-account --image image:tag --spark_conf spark-conf-dir --help "
    exit 1
    ;;
    *)    # action option
    ACTION=$1
    shift 1 # past argument
    ;;
esac
done

# The default API server endpoint access from within the cluster pod
K8S_API_SERVER_ENDPOINT=https://kubernetes.default.svc.cluster.local:443

case "$ACTION" in
  start)
    shift 1
    echo "Using Spark configuration at ${SPARK_CONF}"
    # create spark configmap for Spark conf directory
    kubectl create configmap spark-client-conf --from-file=${SPARK_CONF} --namespace=${K8S_NAMESPACE}
    
    # kubectl apply -f ${WORK_DIR}/templates/spark-client-configmap.yaml
    cat ${WORK_DIR}/templates/spark-client-configmap.yaml | \
    sed "s#\$K8S_API_SERVER_ENDPOINT#${K8S_API_SERVER_ENDPOINT}#g" | \
    sed "s#\$K8S_NAMESPACE#${K8S_NAMESPACE}#g" | \
    sed "s#\$SPARK_DRIVER_SERVICE_ACCOUNT#${SPARK_DRIVER_SERVICE_ACCOUNT}#g" | \
    sed "s#\$SPARK_EXECUTOR_SERVICE_ACCOUNT#${SPARK_EXECUTOR_SERVICE_ACCOUNT}#g" | \
    sed "s#\$CONTAINER_IMAGE#${CONTAINER_IMAGE}#g" | \
    kubectl apply -f -
    
    # create headless service
    # kubectl apply -f ${WORK_DIR}/templates/spark-client-headless-service.yaml
    cat ${WORK_DIR}/templates/spark-client-headless-service.yaml | \
    sed "s#\$K8S_NAMESPACE#${K8S_NAMESPACE}#g" | \
    kubectl apply -f -

    # kubectl apply -f ${WORK_DIR}/templates/spark-client.yaml
    cat ${WORK_DIR}/templates/spark-client.yaml | \
    sed "s#\$K8S_NAMESPACE#${K8S_NAMESPACE}#g" | \
    sed "s#\$SPARK_DRIVER_SERVICE_ACCOUNT#${SPARK_DRIVER_SERVICE_ACCOUNT}#g" | \
    sed "s#\$CONTAINER_IMAGE#${CONTAINER_IMAGE}#g" | \
    kubectl apply -f -
    ;;
  stop)
    shift 1
    kubectl delete pod spark-client --namespace=${K8S_NAMESPACE}
    kubectl delete svc spark-client-headless-service --namespace=${K8S_NAMESPACE}
    kubectl delete configmap spark-client-configmap --namespace=${K8S_NAMESPACE}
    kubectl delete configmap spark-client-conf --namespace=${K8S_NAMESPACE}
    ;;
  *)
    kubectl exec --stdin --tty spark-client -- /bin/bash
    ;;
esac
