#!/usr/bin/env bash

# The cluster config file
cluster_config="./your-cluster-config.yaml"

# The cmd to run Spark Pi job
job_cmd="spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi --conf spark.yarn.submit.waitAppCompletion=false \$SPARK_HOME/examples/jars/spark-examples_2.12-3.2.1.jar 100"

# Start a cluster
cloudtik start $cluster_config -y

# Wait for all workers ready
cloudtik wait-for-ready $cluster_config

# Submit a job and wait for the spark job to finish
cloudtik exec $cluster_config "$job_cmd" --job-waiter=spark

# Stop the cluster
cloudtik stop $cluster_config -y

# The above can also be done with a single exec command
# cloudtik exec $cluster_config "$job_cmd" --start --wait-for-workers  --job-waiter=spark --stop
