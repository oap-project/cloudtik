# Run TPC-DS performance benchmark for Spark on Cloudtik cluster

## 1. Create a new Cloudtik cluster with TPC-DS toolkit
To generate data and run TPC-DS benchmark on Cloudtik cluster, some tools must be installed in advance.
You have several options to do this.

### Option 1: Use a CloudTik Spark runtime image with TPC-DS toolkit installed (Recommended)
In your cluster config under docker key, configure the Spark runtime image with TPC-DS toolkit installed.

```buildoutcfg

docker:
    image: "cloudtik/spark-runtime-tpcds:nightly"

```

This method is preferred as the toolkit is precompiled and installed without impacting cluster starting time.

### Option 2: Use bootstrap commands to compile and install the TPC-DS toolkit
We provide an installation script to simplify the installation of these dependencies.
You only need to add the following bootstrap_commands in the cluster configuration file when you start a cluster.
```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/bootstrap-benchmark.sh &&
        bash ~/bootstrap-benchmark.sh  --workload=tpcds
```
Please note that the toolkit compiling usually takes a long time which will make the cluster ready time much longer than usual.

### Option 3: Use exec commands to run compile and install the TPC-DS toolkit on all nodes
If you cluster already started, you can run the compiling and installing command on all nodes to achieve the same.
```buildoutcfg

cloudtik exec your-cluster-config.yaml "wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/bootstrap-benchmark.sh && bash ~/bootstrap-benchmark.sh --workload=tpcds" --all-nodes

```

Please note that the toolkit compiling usually takes a long time.
You may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't know its completion.

## 2. Generate data
Use "cloudtik status your-cluster-config.yaml" to check the all workers are in ready (update-to-date) status.
If workers are not ready, even you submit a job, the job will still in pending for lack of workers.

We provided the datagen scala script **[tpcds-datagen.scala](./scripts/tpcds-datagen.scala)** for you to generate data in different scales.
Execute the following command to submit and run the datagen script on the cluster,
```buildoutcfg
cloudtik submit your-cluster-config.yaml $CLOUTIK_HOME/tools/benchmarks/spark/scripts/tpcds-datagen.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```
Replace the cluster configuration file, the paths, spark.driver.scale, spark.driver.fsdir values in the above command for your case.

The above command will submit and run the job in foreground and possible need a long time.
you may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't get the command result.
Please refer to [CloudTik Submitting Jobs](https://cloudtik.readthedocs.io/en/latest/UserGuide/AdvancedTasks/submitting-jobs.html) for
the details for run job in background.

## 3. Run TPC-DS power test

We provided the power test scala script **[tpcds-power-test.scala](./scripts/tpcds-power-test.scala)** for users to run TPC-DS power test with Cloudtik cluster.
Execute the following command to submit and run the power test script on the cluster,
```buildoutcfg
cloudtik submit your-cluster-config.yaml $CLOUTIK_HOME/tools/benchmarks/spark/scripts/tpcds-power-test.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --conf spark.sql.shuffle.partitions="\$[\$(cloudtik head info --worker-cpus)*2]" --conf spark.driver.iterations=1 --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```
Replace the cluster configuration file, the paths, spark.driver.scale, spark.driver.fsdir, spark.driver.iterations values in the above command for your case. 

Just like data gen, you may need to run the command with --tmux option for background execution.

When the test is done, you have two options to get the query time results:
1. The query time results will be printed at the end.
2. The query time results will be saved to the configured storage with following location pattern:
"${fsdir}/shared/data/results/tpcds_${format}/${scaleFactor}/"
(replace the fsdir and scaleFactor with the value when submitting job. if 'format' is not specified, it defaults to 'parquet')
You can get the saved file using hadoop command after you attached to the cluster.
