# Running Local Cluster
If you have multiple local nodes and want to run only a single cluster with these nodes,
you can use local provider.

Please follow the below steps to prepare and run a single local cluster.

- System requirements
- Start cluster
- Advanced configurations

## System requirements
The participating nodes need Ubuntu 20.04 or later.

## Start cluster
For local provider, the workspace is degraded to a concept and no needed.

You just need to prepare the cluster configuration
with the information of nodes in the provider section.

For example,

```buildoutcfg
# A unique identifier for the cluster.
cluster_name: example

# The workspace name will be always "default"

# Cloud-provider specific configuration.
provider:
    type: local

    # List of worker nodes with the ip and its node type defined in the above list
    # The local node will be included by default.
    # If you want running on a specific IP of local node, you can also list in this nodes list.
    nodes:
        - ip: node_1_ip
        - ip: node_2_ip
        - ip: node_3_ip

auth:
    # The user is current user with sudo privilege on local host
    ssh_user: ubuntu
    # The private key to SSH to all nodes include local node
    ssh_private_key: ~/.ssh/id_rsa

available_node_types:
    worker.default:
        # The minimum number of worker nodes to launch.
        min_workers: 3

runtimes:
    types: [hdfs, metastore, spark]
```

Execute the following command to start the cluster:
```
cloudtik start /path/to/your-cluster-config.yaml
```

## Advanced configurations
With the above cluster configuration, a default instance type is used for all the nodes.
If the participating nodes are not the same, you can define instance types for specific nodes.

For example,

```buildoutcfg
# A unique identifier for the cluster.
cluster_name: example

# The workspace name will be always "default"

# Cloud-provider specific configuration.
provider:
    type: local

    # List of nodes with the ip and its node type defined in the above list
    nodes:
        - ip: node_1_ip
          # Should be one of the instance types defined in instance_types
          instance_type: my_instance_type
        - ip: node_2_ip
          instance_type: my_instance_type
        - ip: node_3_ip
          instance_type: my_instance_type

    # Define instance types for node. If all nodes are the same with local node
    # You can define a single one or simply don't define and specify instance_type
    instance_types:
        your_instance_type:
            # Specify the resources of this instance type.
            CPU: 4  # number-of-cores
            memory: 4G # memory size, for example 1024M, 1G
auth:
    # The user is current user with sudo privilege on local host
    ssh_user: ubuntu
    # The private key to SSH to all nodes include local node
    ssh_private_key: ~/.ssh/id_rsa

available_node_types:
    worker.default:
        node_config:
            instance_type: my_instance_type
        # The minimum number of worker nodes to launch.
        min_workers: 3
```
