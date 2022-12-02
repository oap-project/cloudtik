# Run TPCx-AI performance benchmark for Spark on Cloudtik cluster

## 1. Create a new Cloudtik cluster with TPCx-AI toolkit
To generate data and run TPCx-AI benchmark on Cloudtik cluster, some tools must be installed in advance.
You have several options to do this.

### Option 1: Use a CloudTik Spark ML runtime image with TPCx-AI toolkit installed (Recommended)
In your cluster config under docker key, configure the Spark runtime image with TPCx-AI toolkit installed.

```buildoutcfg

docker:
    image: "cloudtik/spark-ml-runtime-benchmark:nightly"

```

This method is preferred as the toolkit is installed without impacting cluster starting time.

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

## 2. Generate data for Deep Learning cases
Use "cloudtik status your-cluster-config.yaml" to check the all workers are in ready (update-to-date) status.
If workers are not ready, even you submit a job, the job will still in pending for lack of workers.

Execute the following command to run the datagen script on the cluster,
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase {DATA_GENERATION, LOADING} -sf 1 -c $TPCx_AI_HOME_DIR/driver/config/default-spark.yaml' -uc {2,5,9}
```
Replace the cluster configuration file, scale factor(-sf), useCase(-uc) values in the above command for your case. 
Please note that TPCx-AI supports three deep learing cases(useCase02, useCase05, useCase09), you can seperately generated data for any cases.

The above command will submit and run the job in foreground and possible need a long time.
you may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't get the command result.
Please refer to [CloudTik Submitting Jobs](https://cloudtik.readthedocs.io/en/latest/UserGuide/AdvancedConfigurations/submitting-jobs.html) for
the details for run job in background.

## 3. Run TPCx-AI Deep Learing cases

To run deep learning cases you need to provide custom benchmark configuration file which should be properly defined according to your cluster resources.
There is an example file **[defalut-spark.yaml](tpcx-ai/confs/defalut-spark.yaml)**  and you need to tune the spark parameters for useCase02, useCase05, useCase09.

After you've defined benchmark configuration file, you need to upload this file to head node:
```buildoutcfg
cloudtik rsync-up your-cluster-config.yaml [local path for custom benchmark configuration] [remote path for custom benchmark configuration]
```
Running training stage for useCase02: 
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase TRAINING -sf 1 -c [remote path for custom benchmark configuration] -uc 2'
```
Running serving stage for useCase02:
 ```buildoutcfg
cloudtik exec your-cluster-config.yaml 'cd $TPCx_AI_HOME_DIR && bash bin/tpcxai.sh --phase SERVING -sf 1 -c [remote path for custom benchmark configuration] -uc 2'
```
Replace the cluster configuration file, scale factor(-sf), custom benchmark configuration(-c), useCase number(-uc) valuein the above command for your case. 
