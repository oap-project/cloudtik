# Quickstart AI Application

## 1. Create a new Cloudtik cluster bootstrapped with quickstart AI models
To prepare data and train quickstart models on Cloudtik cluster, you must boostrap the application
to install the necessary tools and application code.
You have several options to do this.

### Option 1: Use a CloudTik oneAPI AI runtime and bootstrap quickstart application (Recommended)
In your cluster config under docker key, configure the oneAPI AI runtime image
and in bootstrap_commands, configure the command for bootstrapping quickstart application.

```buildoutcfg
docker:
    image: "cloudtik/spark-ai-oneapi:nightly"

bootstrap_commands:
    - wget -O ~/bootstrap-quickstart.sh https://raw.githubusercontent.com/oap-project/cloudtik/main/applications/ai/quickstart/scripts/bootstrap-quickstart.sh &&
        bash ~/bootstrap-quickstart.sh
```

### Option 2: Use a CloudTik Spark AI runtime and bootstrap quickstart application
In your cluster config under docker key, configure the Spark AI runtime image
and in bootstrap_commands, configure the command for bootstrapping quickstart application.

```buildoutcfg
```buildoutcfg
docker:
    image: "cloudtik/spark-ai-runtime:nightly"

bootstrap_commands:
    - wget -O ~/bootstrap-quickstart.sh https://raw.githubusercontent.com/oap-project/cloudtik/main/applications/ai/quickstart/scripts/bootstrap-quickstart.sh &&
        bash ~/bootstrap-quickstart.sh
```

### Option 3: Use exec commands to install on all nodes
If you cluster already started, you can run the installing command on all nodes to achieve the same.

If you want to use Intel Extension for PyTorch, run the following command.
If you are using oneAPI AI runtime, you can skip this step.
```buildoutcfg
cloudtik exec your-cluster-config.yaml "wget -O ~/bootstrap-ipex.sh https://raw.githubusercontent.com/oap-project/cloudtik/main/applications/ai/quickstart/scripts/bootstrap-ipex.sh && bash ~/bootstrap-ipex.sh" --all-nodes
```

Run the following command for installing quickstart application.
```buildoutcfg
cloudtik exec your-cluster-config.yaml "wget -O ~/bootstrap-quickstart.sh https://raw.githubusercontent.com/oap-project/cloudtik/main/applications/ai/quickstart/scripts/bootstrap-quickstart.sh && bash ~/bootstrap-quickstart.sh" --all-nodes
```

Please note that the toolkit installing may take some time.
You may need to run the command with --tmux option for background execution
for avoiding terminal disconnection in the middle. And you don't know its completion.

Once the cluster start completed or bootstrap completed,
the quickstart application is installed and configure at '$HOME/applications/quickstart'.

## 2. Prepare data and run training or inference for a specific task.
Now you can run data preparation, training or inference for a supported model.
The quickstart application support the following AI models:
- BERT-large [model name: bert-large](./bin/bert-large)
- DLRM [model name: dlrm](./bin/dlrm)
- Mask R-CNN [model name: maskrcnn](./bin/maskrcnn)
- ResNet-50 [model name: resnet50](./bin/resnet50)
- ResNeXt101 [model name: resnext-32x16d](./bin/resnext-32x16d)
- RNN-T [model name: rnnt](./bin/rnnt)
- SSD-ResNet-34 [model name: ssd-resnet34](./bin/ssd-resnet34)

### Preparing data
Run the following command for preparing data.
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'bash $HOME/applications/quickstart/bin/[model-name]/prepare-data.sh'
```
Replace model-name to one of the model name above.

### Run inference
Run the following command for inference if supported.
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'bash $HOME/applications/quickstart/bin/[model-name]/inference.sh'
```
Replace model-name to one of the model name above.

### Run single node training
Run the following command for training if supported.
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'bash $HOME/applications/quickstart/bin/[model-name]/train.sh'
```
Replace model-name to one of the model name above.

### Run distributed training
Run the following command for training if supported.
```buildoutcfg
cloudtik exec your-cluster-config.yaml 'bash $HOME/applications/quickstart/bin/[model-name]/train-distributed.sh'
```
Replace model-name to one of the model name above.
