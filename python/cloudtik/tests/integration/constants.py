CLUSTER_TIMEOUT = 60 * 10

AWS_BASIC_CLUSTER_CONF_FILE = "example/cluster/aws/example-standard.yaml"
AWS_BASIC_WORKSPACE_CONF_FILE = "example/cluster/aws/example-workspace.yaml"
AZURE_BASIC_CLUSTER_CONF_FILE = "example/cluster/azure/example-standard.yaml"
AZURE_BASIC_WORKSPACE_CONF_FILE = "example/cluster/azure/example-workspace.yaml"
GCP_BASIC_CLUSTER_CONF_FILE = "example/cluster/gcp/example-standard.yaml"
GCP_BASIC_WORKSPACE_CONF_FILE = "example/cluster/gcp/example-workspace.yaml"
HUAWEICLOUD_BASIC_CLUSTER_CONF_FILE = "example/cluster/huaweicloud/example-standard.yaml"
HUAWEICLOUD_BASIC_WORKSPACE_CONF_FILE = "example/cluster/huaweicloud/example-workspace.yaml"

TPCDS_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/tpcds-power-test.scala",
    "script_args": ' --conf spark.driver.fsdir="/tpcds" --jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar'
}

TPC_DATAGEN_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/tpcds-datagen.scala",
    "script_args": ' --conf spark.driver.fsdir="/tpcds" --jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar'
}

KAFKA_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/kafka/scripts/kafka-benchmark.sh",
    "script_args": ""
}

PRESTO_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/presto/scripts/tpcds-tpch-power-test.sh",
    "script_args": " --workload=tpcds --scale=1 --iteration=1"
}

runtime_additional_conf = {
    'runtime': {'types': ['ganglia', 'metastore', 'spark', 'kafka']}}

WORKER_NODES_LIST = [1, 4]
SCALE_CPUS_LIST = [2, 6, 14]
SCALE_NODES_LIST = [1, 2, 4]
