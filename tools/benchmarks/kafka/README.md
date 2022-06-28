# Run kafka-benchmark for Kafka on Cloudtik cluster

## 1. Create a new Cloudtik cluster
To run kafka-benchmark on Cloudtik cluster, you need to download the benchmark script on head node.
You only need to add the following bootstrap_commands in the cluster configuration file.
```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/kafka/scripts/kafka-benchmark.sh
```

## 2. Run Kafka benchmark

Execute the following command to run kafka-benchmark on the cluster:
```buildoutcfg
cloudtik exec your-cluster-config.yaml "bash \$HOME/kafka-benchmark.sh"
```
Replace the cluster configuration file for your case. 
