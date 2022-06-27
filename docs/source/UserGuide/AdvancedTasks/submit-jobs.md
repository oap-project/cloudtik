# Submit Jobs
Cloudtik provides the ability to easily submit tasks to your clusters. Currently Cloudtik can support submit local script files and web script files, supported file types are ***.sh, .scala, .py, .presto.sql, .trino.sql***.



## How to Submit Jobs

Submit a job to cluster to run.

```
cloudtik submit [OPTIONS] /path/to/your-cluster-config.yaml $YOUE_SCRIPT [SCRIPT_ARGS]
```
#### OPTIONS
```
  --stop                          Stop the cluster after the command finishes running.
  --start                         Start the cluster if needed.
  --screen                        Run the command in a screen.
  --tmux                          Run the command in tmux.
  -n, --cluster-name TEXT         Override the configured cluster name.
  --no-config-cache               Disable the local cluster config cache.
  -p, --port-forward INTEGER      Port to forward. Use this multiple times to forward multiple ports.
  --log-style [auto|record|pretty]
                                  If 'pretty', outputs with formatting and
                                  color. If 'record', outputs record-style
                                  without formatting. 'auto' defaults to
                                  'pretty', and disables pretty logging if
                                  stdin is *not* a TTY.
  --log-color [auto|false|true]   Use color logging. Auto enables color
                                  logging if stdout is a TTY.
```  



For example:

    # --smoke-test is an option for experiment.py
    cloudtik submit /path/to/your-cluster-config.yaml experiment.py  --smoke-test


#### Asynchronously submit jobs
    
Cloudtik use **[screen](https://www.gnu.org/software/screen/manual/screen.html)** and **[tmux](https://github.com/tmux/tmux/wiki/Getting-Started)** to support asynchronously run job. You can add `--screen` or `--tmux` option in your command.
```bash
   cloudtik submit --screen /path/to/your-cluster-config.yaml experiment.py  --smoke-test
```
Once the job is submitted, Cloudtik will automatically start a screen session to run your scripts.
This will bring great convenience to running some long-time tasks. Combined with Cloudtik **attach** function, you can easily disconnect or connect to the screen of the task at any time to view the running output of the task.



#### Notes 
1. The script file will be automatically synced/downloaded to the path "~/jobs/".
2. Sometimes your parameters for the script will contain special character like ***|,\\***, it will cause the parameters can not be parsed correctly.
 We suggest you use single quote to avoid these problems. For example:
   ''
```    
    --conf spark.oap.sql.columnar.coreRange='"0-31,64-95|32-63,96-127"'  --jars '$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar'
```  

## Run TPC-DS on Spark cluster

Here is an example of how to use `cloudtik submit` to run TPC-DS on the created cluster.

###### 1. Creating a cluster 

To generate data and run TPC-DS on a cluster, some extra tools need be installed to nodes within cluster setup steps. 
We provide a script to simplify the installation of these dependencies. You only need to add the following bootstrap_commands to the cluster configuration file.

```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/bootstrap-benchmark.sh &&
        bash ~/bootstrap-benchmark.sh  --tpcds
```


###### 2. Generating data

We provided the datagen scala script which can be found from CloudTik's `./tools/spark/benchmark/scripts/tpcds-datagen.scala` for you to generate data in different size.

Execute the following command to submit and run the script of generating data after the cluster's all nodes are ready.

```
cloudtik submit /path/to/your-cluster-config.yaml $CLOUTIK_HOME/tools/spark/benchmark/scripts/tpcds-datagen.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```

`$CLOUTIK_HOME/tools/spark/benchmark/scripts/tpcds-datagen.scala` is the script's location on your working node.

`spark.driver.scaleFactor=1` is to generate 1 GB data, you can change it by case. 

`spark.driver.fsdir="s3a://s3_bucket_name"` is to specify S3 bucket name, change it to your bucket link of cloud storage.

`--jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar` specifies the default path of spark-sql-perf jar when the cluster nodes are set up, just leave it untouched.


###### 3. Run TPC-DS power test

We provided the power test scala script which can be found from CloudTik's `./tools/spark/benchmark/scripts/tpcds-power-test.scala` for users to run TPC-DS power test with Cloudtik cluster.

Execute the following command to submit and run the power test script on the cluster,

```buildoutcfg
cloudtik submit /path/to/your-cluster-config.yaml $CLOUTIK_HOME/tools/spark/benchmark/scripts/tpcds-power-test.scala --conf spark.driver.scaleFactor=1 --conf spark.driver.fsdir="s3a://s3_bucket_name" --conf spark.sql.shuffle.partitions=\$(cloudtik head info --worker-cpus) --conf spark.driver.iterations=1 --jars \$HOME/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar
```