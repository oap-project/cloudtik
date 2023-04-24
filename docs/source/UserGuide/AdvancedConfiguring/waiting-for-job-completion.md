# Waiting for Job Completion
When you are using submit or exec to run jobs in the cluster, you may need to wait for
the job completion such as when you want to stop the cluster when the job finished.

For foreground job, you don't need any additional mechanisms as the job is done when the script completes.
When you are submitting or executing job with --tmux or --screen, or when you are running a cluster mode
Spark job which is running at the background, you need additional job waiter to do completion check.

CloudTik designed the job waiter mechanisms for you to achieve the capability and provides you the
way to specify your own job waiters.

## Use job waiter in submit or exec command
When you need to wait for an async job completion, you can specify --job-waiter parameter
with a job waiter name.

```
cloudtik exec your-cluster-config.yaml "sleep 10" --job-waiter=job-waiter-name
```

When a job waiter is specified, CloudTik will call wait_for_completion for the job waiter
to wait for its completion.

## Built-in job waiters
CloudTik implemented a few useful built-in job waiters. 

### Tmux job waiter
When you run a script using --tmux in the background to avoid broken session problem,
you can use the built-in Tmux job waiter to wait for a background tmux session to complete.
The name of Tmux job waiter to use is "tmux". For example,

```
cloudtik exec your-cluster-config.yaml "sleep 10" --tmux --job-waiter=tmux
```

The command above will sleep 10 seconds in a background tmux session and wait for its completion.

### Screen job waiter
When you run a script using --screen in the background to avoid broken session problem,
you can use the built-in Screen job waiter to wait for a background screen session to complete.
The name of Screen job waiter to use is "screen". For example,

```
cloudtik exec your-cluster-config.yaml "sleep 10" --screen --job-waiter=screen
```

The command above will sleep 10 seconds in a background screen session and wait for its completion.

### job waiter chain
There are cases that you need to wait for multiple jobs which were results from executing a script.
CloudTik designed a built-in job waiter to help to chain multiple job waiter for such complex cases.
The name of job waiter Chain to use is "chain" while you can specify a list of chained job waiters in the format
of chain[job-waiter-1, job-waiter-2, ...].
For example,

```
cloudtik exec your-cluster-config.yaml "your-job-script.sh" --tmux --job-waiter=chain[tmux, your-waiter-name]
```

## Runtime job waiters
The runtime is designed to provide job waiter for waiting for its job completion if there is a need.
If one runtime provide a job waiter, you can use the runtime name as the job waiter name to use
the job waiter provide by the runtime. For example, "spark" job waiter as described in the following section.

### Spark job waiter
Spark runtime provides "spark" job waiter to wait for all Spark applications to complete in a Spark cluster.
Internally, we use YARN to check all the YARN application's completion.

You need Spark job waiter only when you submit an async Spark job which means a YARN cluster mode application.

## Useful examples

### Spark job example
Assume you are submitting a Spark job with cluster mode and also specifying --tmux parameter when executing,
And you want to stop the cluster after the job completed.

```
cloudtik exec your-cluster-config.yaml \
    "spark-submit --master yarn --deploy-mode cluster --name spark-pi --class org.apache.spark.examples.SparkPi \$SPARK_HOME/examples/jars/spark-examples.jar" \
    --tmux --job-waiter=chain[tmux, spark] --stop
```

Because you used --tmux parameter, you need tmux job waiter in the chain to wait for the tmux session completion
which means the spark job submitting completion. From this point, we need a Spark job waiter to wait for the submitted
application completion. Specifying --stop to stop the cluster after the job completed.

### Deep learning job example
Assume you are submitting a deep learning job to a machine learning cluster to do AI training,

You may start with this command, for example,
```
cloudtik submit your-cluster-config.yaml \
    ./cloudtik/examples/ml/jobs/keras/mnist-keras-spark-horovod-hyperopt-mlflow.py
```
You will find that the training job will run quite a long time.
And you want it to run in background, so you submit the job as the following,
```
cloudtik submit your-cluster-config.yaml \
    ./cloudtik/examples/ml/jobs/keras/mnist-keras-spark-horovod-hyperopt-mlflow.py \
    --tmux
```
Since the training job output a lot of information.
And you want to check the results afterwards, so you submit the job as the following,
```
cloudtik submit your-cluster-config.yaml \
    ./cloudtik/examples/ml/jobs/keras/mnist-keras-spark-horovod-hyperopt-mlflow.py \
    --tmux --job-log
```
The --job-log parameter will redirect the job script output to the log file at ~/user/logs.
