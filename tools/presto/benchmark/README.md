# Run TPC-DS benchmark for Presto on Cloudtik cluster

## 1. Create a new Cloudtik cluster
To run TPC-DS or TPC-H on Cloudtik cluster, you need to download the benchmark script on head node.
You only need to add the following bootstrap_commands in the cluster configuration file.
```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/presto/benchmark/scripts/tpcds-tpch-power-test.sh
```

## 2. Run TPC-DS power test

Execute the following command to run TPC-DS on the cluster:
```buildoutcfg
cloudtik exec your-cluster-config.yaml "bash \$HOME/tpcds-tpch-power-test.sh --workload=tpcds --scale=1 --iteration=2"
```
Replace the cluster configuration file for your case. 

## 3. Run TPC-H power test

Execute the following command to run TPC-DS on the cluster:
```buildoutcfg
cloudtik exec your-cluster-config.yaml "bash \$HOME/tpcds-tpch-power-test.sh --workload=tpch --scale=1 --iteration=2"
```
Replace the cluster configuration file for your case. 