# Creating Cluster

The cluster configuration is defined within a YAML file. It will be used by CloudTik to launch head node, and its cluster
controller on head node to launch worker nodes.

CloudTik provides cluster configuration yaml examples, which are located under CloudTik's `example/cluster/` directory.
 
Please follow instructions below to customize your cluster configuration.

## Execution Mode

Choosing between host mode or container mode is simple with 'enabled' option under 'docker' config.

The following config chooses to run on host mode:
```
# Turn on or off container by setting "enabled" to True or False.
docker:
    enabled: False

```

The following config chooses to run on container mode:

```
# Enable container
docker:
    enabled: True
```

There are a few other advanced options for container mode.
For details, please refer to [Advanced Tasks: Configuring Container Mode](./AdvancedTasks/configuring-container-mode.md)

## Controlling the Number of Workers

The minimum number of worker nodes to launch, and the default number is 1. You can change it according to your use case
to overwrite default value as below, which sets minimum number of worker nodes to 3.

On AWS or Azure:
```
available_node_types:
    worker.default:
        min_workers: 3
```

On GCP:
```
available_node_types:
    worker-default:
        min_workers: 3
```

## Choosing Runtimes for Cluster

Without extra configuration, CloudTik creates a cluster with the Ganglia and Spark runtimes by default.
You can use 'types' configure key under 'runtime' to configure the list of runtimes for the cluster.

The following example will start a cluster with Ganglia, HDFS as default storage and Spark.

```
runtime:
    types: [ganglia, hdfs, spark]
```

For more detailed information of each runtime, please refer to [Reference: Runtimes](../Reference/runtimes.md)

## Using Bootstrap Commands to Customize Setup
You can specify a list of commands in the 'bootstrap_commands' in cluster config
to run customized setup commands.

```
# Commands running after common setup commands
bootstrap_commands: ['your shell command 1', 'your shell command 2']
```

For example, If you want to add more integrations to your clusters within the setup steps, such as adding required tools to run TPC-DS, 
please add the following `bootstrap_commands` section to your cluster yaml file, which will install TPC-DS and required packages
with specified scripts to set up nodes after all common setup commands finish.

```buildoutcfg

bootstrap_commands:
    - wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/spark/benchmark/scripts/bootstrap-benchmark.sh &&
        bash ~/bootstrap-benchmark.sh  --tpcds
```
CloudTik allows user to custom as mang commands as CloudTik itself does.
For more advanced commands customization, please refer to [Advanced Tasks: Using Custom Commands](./AdvancedTasks/using-custom-commands.md)

## Mounting Files or Directories to Nodes

To mount files or directories to each node when cluster starting up, add the following to cluster configuration file.

```
# Files or directories to copy to the head and worker nodes. The format is a
# dictionary from REMOTE_PATH: LOCAL_PATH, e.g.

file_mounts: {
#    "/path1/on/remote/machine": "/path1/on/local/machine",
#    "/path2/on/remote/machine": "/path2/on/local/machine",
     "~/.ssh/id_rsa.pub": "~/.ssh/id_rsa.pub"
}
```

## Using Templates to Simplify Node Configuration
CloudTik designs a templating structure to allow user to reuse a standard configurations.
A template defines a typical or useful configurations for node such as the instance type and disk configurations.
By inheriting from a template, you get these configurations by default.
You can use the keyword 'from' to specify the template you want to inherit from.

For example, the following instruction at the beginning of the cluster config declares
the inheritance from a template called 'aws/standard' which is defined as part of CloudTik
within `python/cloudtik/templates` folder.

```
from: aws/standard
```

For a list of CloudTik defined system templates, please refer to `python/cloudtik/templates` directory
under the cloudtik pip installation.

For more details as to templates, please refer to [Advanced Tasks: Using Templates](./AdvancedTasks/using-templates.md)


Once the cluster configuration is defined and CloudTik is installed, you can use the following commands to create a cluster with CloudTik

```
$ cloudtik start /path/to/your-cluster-config.yaml -y
```

After that, you can also use the CloudTik commands to check the status or manage the cluster.

Please refer to [Managing Cluster](./managing-cluster.md) for detailed instructions.
