#!/usr/bin/env bash

# The cluster config file: change to your file
cluster_config="./your-cluster-config.yaml"

# The python job: change to your job file
job_file="./cloudtik/example/ml/jobs/spark-mlflow-hyperopt-horovod-tensorflow.py"

# Start a cluster
cloudtik start $cluster_config -y

# Wait for all workers ready
cloudtik wait-for-ready $cluster_config

# Submit a python job and run in a background tmux session and wait the tmux session to complete
cloudtik submit $cluster_config $job_file --tmux --job-waiter=tmux

# Stop the cluster
cloudtik stop $cluster_config -y

# The above can also be done with a single submit command
# cloudtik submit $cluster_config $job_file --start --wait-for-workers --tmux --job-waiter=tmux --stop
