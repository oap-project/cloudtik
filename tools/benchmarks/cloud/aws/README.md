# Run TPC-DS power test with SF1000 for Spark on AWS

We provide some scripts which can help you quickly setup a cluster with 1 m5.4xlarge instance as head and three m5.16xlarge instances as workers on AWS and use S3 to store data.
You can easily generate 1TB data for TPC-DS and run TPC-DS power test with vanilla spark or **[gazelle_plugin](https://raw.githubusercontent.com/oap-project/oap-tools/master/integrations/oap/emr/bootstrap_oap.sh)**.

## 1. Create and configure a YAML file for workspace
We provide a example workspace yaml file **[aws-workspace.yaml](./aws-workspace.yaml)**, 
and you can modify this example file according to your requirements. 

## 2. Create a workspace on AWS
We provide the script **[aws-resource.sh](./scripts/aws-resource.sh)** to create or delete a workspace. 

```buildoutcfg
# Create a workspace on AWS
bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-resource.sh --action create-workspace --config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-workspace.yaml --yes

# Delete the workspace
# bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-resource.sh --action delete-workspace --config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-workspace.yaml --yes
```

## 3. Create and configure a YAML file for cluster
We provide a example cluster yaml file **[aws-large-cluster-with-s3.yaml](./aws-large-cluster-with-s3.yaml)**, 
and you need to modify this yaml file according to your workspace configuration. 

## 4. Create a cluster on AWS
We provide the script **[aws-resource.sh](./scripts/aws-resource.sh)** to start or stop a cluster. 

```buildoutcfg
# Start a cluster on AWS
bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-resource.sh --action start-cluster --config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-large-cluster-with-s3.yaml --yes

# Stop the cluster
# bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-resource.sh --action stop-cluster --config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-large-cluster-with-s3.yaml --yes
```

## 5. Generate data
We provided the shell script **[aws-benchmark.sh](./scripts/aws-benchmark.sh)** for you to generate data in different scales.
```buildoutcfg
# Generate TPC-DS SF1000 data
bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-benchmark.sh --action generate-data --cluster_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-large-cluster-with-s3.yaml --workspace_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-workspace.yaml --scale_factor 1000
```

## 6. Run TPC-DS power test with vanilla spark
We provided the shell script **[aws-benchmark.sh](./scripts/aws-benchmark.sh)** for you to easily run TPC-DS power test with vanilla spark.
```buildoutcfg
# Run TPC-DS power test with SF1000 for 1 round 
bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-benchmark.sh --action run --cluster_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-large-cluster-with-s3.yaml --workspace_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-workspace.yaml --scale_factor 1000 --iteration=1 --baseline
```

## 7. Run TPC-DS power test with gazelle_plugin
We provided the shell script **[aws-benchmark.sh](./scripts/aws-benchmark.sh)** for you to easily run TPC-DS power test with **[gazelle_plugin](https://raw.githubusercontent.com/oap-project/oap-tools/master/integrations/oap/emr/bootstrap_oap.sh)**. You must provide aws_access_key_id and aws_secret_access_key for gazelle_plugin to access S3.
```buildoutcfg
# Run TPC-DS power test with SF1000 for 1 round
bash $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/scripts/aws-benchmark.sh --action run --cluster_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-large-cluster-with-s3.yaml --workspace_config $CLOUDTIK_HOME/tools/benchmarks/cloud/aws/aws-workspace.yaml --scale_factor 1000 --iteration=1 --aws_access_key_id=[key_id] --aws_secret_access_key=[key]
```
