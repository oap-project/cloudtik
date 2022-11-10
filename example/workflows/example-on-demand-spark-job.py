from cloudtik.core.api import Cluster

# The cluster config file: change to your file
cluster_config = "./your-cluster-config.yaml"

# The cmd to run: change to your cmd
job_cmd = "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi --conf spark.yarn.submit.waitAppCompletion=false $SPARK_HOME/examples/jars/spark-examples_2.12-3.2.1.jar 100"

# Create a new cluster client object to operate
cluster = Cluster(cluster_config)

# Start a cluster
cluster.start()

# Wait for all workers ready
cluster.wait_for_ready()

# Exec a job and wait for the spark job to finish
cluster.exec(job_cmd, job_waiter="spark")

# Stop the cluster
cluster.stop()

# The above can also be done with a single exec call
# cluster.exec(job_cmd, start=True, wait_for_workers=True, job_waiter="spark", stop=True)
