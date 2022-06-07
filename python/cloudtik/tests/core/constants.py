
CLUSTER_TIMEOUT = 60 * 5

BASIC_WORKSPACE_CONF_FILES = ["example/cluster/aws/example-workspace.yaml",
                              "example/cluster/azure/example-workspace.yaml",
                              "example/cluster/gcp/example-workspace.yaml"]
BASIC_CLUSTER_CONF_FILES = ["example/cluster/aws/example-standard.yaml",
                            "example/cluster/azure/example-standard.yaml",
                            "example/cluster/gcp/example-standard.yaml"]
AWS_BASIC_CLUSTER_CONF_FILES = ["example/cluster/aws/example-standard.yaml"]

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