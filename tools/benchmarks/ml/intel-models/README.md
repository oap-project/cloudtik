# Run IntelAI Models benchmark on Cloudtik cluster

## 1. Create a new Cloudtik cluster with IntelAI Models
To prepare data and run Models on Cloudtik cluster, some tools must be installed in advance.
You have several options to do this.

### Option 1: Use a CloudTik Spark ML runtime image with IntelAI Models installed (Recommended)
In your cluster config under docker key, configure the ML runtime image with IntelAI Models installed.

```buildoutcfg
docker:
    image: "cloudtik/spark-ml-runtime-benchmark:nightly"
```

### Option 2: Use bootstrap commands to install the IntelAI Models
We provide an installation script to simplify the installation of these dependencies.
You only need to add the following bootstrap_commands in the cluster configuration file when you start a cluster.
```buildoutcfg
bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/intel-models/scripts/bootstrap-models.sh &&
        bash ~/bootstrap-models.sh
```

### Option 3: Use exec commands to install the IntelAI Models on all nodes
If you cluster already started, you can run the installing command on all nodes to achieve the same.
```buildoutcfg
cloudtik exec your-cluster-config.yaml "wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/intel-models/scripts/bootstrap-models-benchmark.sh && bash ~/bootstrap-models.sh" --all-nodes
```

Please note that the toolkit installing may take a long time.
You may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't know its completion.

## 2. Prepare data and run Models benchmark for a specific workload
CloudTik Models benchmark support the following specific Models cases:
- Bert ([README](./bert/README.md))
- ResNet ([README](./resnet/README.md))
- DLRM ([README](./dlrm/README.md))

Refer to the corresponding README for detail instructions as to
preparing and running the case training or inference.