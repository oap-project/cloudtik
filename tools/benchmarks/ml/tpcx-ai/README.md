# Run TPCx-AI performance benchmark for Spark on Cloudtik cluster

## 1. Create a new Cloudtik cluster with TPCx-AI toolkit
To generate data and run TPCx-AI benchmark on Cloudtik cluster, some tools must be installed in advance.
You have several options to do this.

### Option 1: Use a CloudTik Spark ML runtime image with TPCx-AI toolkit installed (Recommended)
In your cluster config under docker key, configure the ML runtime image with TPCx-AI toolkit installed.

```buildoutcfg
docker:
    image: "cloudtik/spark-ml-runtime-benchmark:nightly"
```

### Option 2: Use bootstrap commands to install the TPCx-AI toolkit
We provide an installation script to simplify the installation of these dependencies.
You only need to add the following bootstrap_commands in the cluster configuration file when you start a cluster.
```buildoutcfg
bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/bootstrap-tpcx-ai-benchmark.sh &&
        bash ~/bootstrap-tpcx-ai-benchmark.sh
```

### Option 3: Use exec commands to install the TPCx-AI toolkit on all nodes
If you cluster already started, you can run the installing command on all nodes to achieve the same.
```buildoutcfg
cloudtik exec your-cluster-config.yaml "wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/bootstrap-tpcx-ai-benchmark.sh && bash ~/bootstrap-tpcx-ai-benchmark.sh" --all-nodes
```

Please note that the toolkit installing usually takes a long time.
You may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't know its completion.

## 2. Update the ip of all workers for TPCx-AI
After the cluster has been created, you need to run the following command to update the ip of all workers for TPCx-AI:
```buildoutcfg
cloudtik exec your-cluster-config.yaml "cloudtik head worker-ips > ~/runtime/benchmark-tools/tpcx-ai/nodes"
```

## 3. Generate data for Deep Learning cases
Use "cloudtik status your-cluster-config.yaml" to check the all workers are in ready (update-to-date) status.
If workers are not ready, even you submit a job, the job will still in pending for lack of workers.

Execute the following command to run the datagen script on the cluster:
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'source ~/runtime/benchmark-tools/tpcx-ai/setenv.sh && cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase {DATA_GENERATION,LOADING} -sf 1 -c $TPCx_AI_HOME_DIR/driver/config/default-spark.yaml -uc {2,5,9}'
```
Replace the cluster configuration file, scale factor(-sf), useCase(-uc) values in the above command for your case. 
Please note that TPCx-AI supports three deep learning cases(useCase02, useCase05, useCase09), you can seperately generated data for any cases.

The above command will submit and run the job in foreground and possible need a long time.
you may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't get the command result.
Please refer to [CloudTik Submitting Jobs](https://cloudtik.readthedocs.io/en/latest/UserGuide/AdvancedConfigurations/submitting-jobs.html) for
the details for run job in background.

## 4. Run TPCx-AI Deep Learing cases

To run deep learning cases you need to provide custom benchmark configuration file which should be properly defined according to your cluster resources.
There is an example file **[default-spark.yaml](tpcx-ai/confs/default-spark.yaml)**  and you need to tune the spark parameters for useCase02, useCase05, useCase09.

After you've defined benchmark configuration file, you need to upload this file to head node:
```buildoutcfg
cloudtik rsync-up your-cluster-config.yaml [local path for custom benchmark configuration] [remote path for custom benchmark configuration]
```
Running training stage for useCase02: 
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'source ~/runtime/benchmark-tools/tpcx-ai/setenv.sh && cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase TRAINING -sf 1 -c [remote path for custom benchmark configuration] -uc 2'
```
Running serving stage for useCase02:
 ```buildoutcfg
cloudtik exec your-cluster-config.yaml 'source ~/runtime/benchmark-tools/tpcx-ai/setenv.sh && cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase SERVING -sf 1 -c [remote path for custom benchmark configuration] -uc 2'
```
Running training and serving stage for useCase02, useCase05, useCase09:
 ```buildoutcfg
cloudtik exec your-cluster-config.yaml 'source ~/runtime/benchmark-tools/tpcx-ai/setenv.sh && cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase {TRAINING,SERVING} -sf 1 -c [remote path for custom benchmark configuration] -uc {2,5,9}'
```
The final result will be similar to the output below. Each line will display the time consumed by each stage of each case, and the time unit is seconds.
```buildoutcfg
========== RESULTS ==========
phase_name  Phase.SERVING_1  Phase.TRAINING_1
use_case                                     
2                   153.826           430.651
5                    91.367           200.453
9                    93.398           246.164
```
