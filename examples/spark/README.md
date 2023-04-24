# Running Spark Pi Example

It is quite straight forward for running Spark built-in PI example
once you have a CloudTik with Spark runtime started.

Simple run the following command on your client machine:
```
cloudtik exec ./your-cluster-config.yaml "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi --conf spark.yarn.submit.waitAppCompletion=false \$SPARK_HOME/examples/jars/spark-examples.jar 12345" --job-waiter=spark
```
This will submit a Spark Pi job to the cluster and running in the background.
This command will wait for the job to finish by specifying Spark job waiter "--job-waiter=spark".

If you want to run the job fully foreground, you can execute:
```
cloudtik exec ./your-cluster-config.yaml "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi \$SPARK_HOME/examples/jars/spark-examples.jar 12345"
```
