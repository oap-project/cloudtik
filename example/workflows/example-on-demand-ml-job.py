from cloudtik.core.api import Cluster

# The cluster config file: change to your file
cluster_config = "./your-cluster-config.yaml"

# The python job: change to your job file
job_file = "./cloudtik/example/ml/jobs/keras/mnist-keras-spark-horovod-run-hyperopt-mlflow.py"

# Create a new cluster client object to operate
cluster = Cluster(cluster_config)

# Start a cluster
cluster.start()

# Wait for all workers ready
cluster.wait_for_ready()

# Submit a job and wait for the spark job to finish
cluster.submit(job_file, tmux=True, job_waiter="tmux")

# Stop the cluster
cluster.stop()

# The above can also be done with a single submit call
# cluster.submit(job_file, start=True, wait_for_workers=True, tmux=True, job_waiter="tmux", stop=True)
