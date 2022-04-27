# Run TPC-DS performance benchmark for Spark on Cloudtik cluster

## 1. Create a new Cloudtik cluster
To generate data and run TPC-DS benchmark on Cloudtik cluster, some tools must be installed in advance.
We provide an installation script to simplify the installation of these dependencies. You only need to add the following bootstrap_commands in the cluster configuration file.
```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/bootstrap-benchmark.sh &&
        bash ~/bootstrap-benchmark.sh  --tpcds
```

In addition, if you also want to install OAP packages by Conda on CloudTik clusters, please also add the following to `bootstrap_commands` section of cluster yaml file as above.

```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/bootstrap_oap.sh &&
        bash ~/bootstrap_oap.sh
```



## 2. Generate data

We provided the datagen scala script **[tpcds-datagen.scala](./scripts/tpcds-datagen.scala)** for you to generate data in different scales.
Execute the following command to submit and run the datagen script on the cluster,
```buildoutcfg
cloudtik submit your-cluster-config.yaml $CLOUTIK_HOME/tools/spark/benchmark/scripts/tpcds-datagen.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```
Replace the cluster configuration file, the paths, spark.driver.scale, spark.driver.fsdir values in the above command for your case.

## 3. Run TPC-DS power test

We provided the power test scala script **[tpcds-power-test.scala](./scripts/tpcds-power-test.scala)** for users to run TPC-DS power test with Cloudtik cluster.
Execute the following command to submit and run the power test script on the cluster,
```buildoutcfg
cloudtik submit your-cluster-config.yaml $CLOUTIK_HOME/tools/spark/benchmark/scripts/tpcds-power-test.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --conf spark.sql.shuffle.partitions=\$(cloudtik head info --worker-cpus) --conf spark.driver.iterations=1 --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```
Replace the cluster configuration file, the paths, spark.driver.scale, spark.driver.fsdir, spark.driver.iterations values in the above command for your case. 
