CLUSTER_TIMEOUT = 60 * 5

AWS_BASIC_CLUSTER_CONF_FILE = "example/cluster/aws/example-standard.yaml"
AWS_BASIC_WORKSPACE_CONF_FILE = "example/cluster/aws/example-workspace.yaml"
AZURE_BASIC_CLUSTER_CONF_FILE = "example/cluster/azure/example-standard.yaml"
AZURE_BASIC_WORKSPACE_CONF_FILE = "example/cluster/azure/example-workspace.yaml"
GCP_BASIC_CLUSTER_CONF_FILE = "example/cluster/gcp/example-standard.yaml"
GCP_BASIC_WORKSPACE_CONF_FILE = "example/cluster/gcp/example-workspace.yaml"

TPCDS_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/tpcds-power-test.scala",
    "script_args": "--jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar"
}

TPC_DATAGEN_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/tpcds-datagen.scala",
    "script_args": "--jars /home/cloudtik/runtime/benchmark-tools/spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar"
}

KAFKA_BENCHMARK = {
    "script_file": "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/kafka/benchmark/scripts/kafka-benchmark.sh",
    "script_args": ""
}

runtime_additional_conf = {
    'setup_commands': 'wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/bootstrap-benchmark.sh &&bash ~/bootstrap-benchmark.sh  --tpcds ',
    'runtime': {'types': ['ganglia', 'metastore', 'spark', 'kafka']}}

WORKER_NODES_LIST = [1, 2, 4, 6]
SCALE_CPUS_LIST = [2, 6, 8, 14]
SCALE_NODES_LIST = [1, 2, 4, 6]
