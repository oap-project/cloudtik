# Running Virtual Clusters on Single Node
The simplest way to play or develop with CloudTik is running virtual clusters
on a single node. This capability is provided by the virtual provider which
using local docker containers to provide the similar functionality as any public
cloud VMs.

With virtual provider, user can even isolate multiple clusters with the workspace
with clusters of each workspace on a different docker bridge network.

The workspace and cluster steps are the same as you running on public clouds.
But for running virtual clusters, user needs only very minimum configurations.

Please follow the below steps to prepare and run virtual clusters.

- System requirements
- Install and configure docker
- Create workspace
- Start cluster in workspace

## System requirements
Ubuntu 20.04 or later.
 
## Install and configure docker
The virtual provider needs docker be installed and configured running docker commands without sudo.

For docker installation, please refer to `dev/install-docker.sh` to install Docker on Ubuntu Linux.

To configure running docker without sudo, execute the following command:

```buildoutcfg
sudo addgroup --system docker;
sudo usermod -aG docker $USER;
sudo systemctl restart docker -f;
```
After executed these commands,
logout and login the shell to make the group updates taking effect.

## Create workspace
With virtual provider, you can run clusters in different workspace (bridge network).

The workspace configuration is very simple.

```buildoutcfg
# A unique identifier for the workspace.
workspace_name: example-workspace

# For virtual, a workspace is isolated by a docker bridge on host
provider:
    type: virtual
```

Execute the following command to create the workspace:
```
cloudtik workspace create /path/to/your-workspace-config.yaml
```

## Start cluster

The cluster configuration for virtual provider is also quite simple with
a resource specification of head and worker instance types.

For example,

```buildoutcfg
# A unique identifier for the cluster.
cluster_name: example

# The workspace name
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: virtual

auth:
    ssh_user: ubuntu

available_node_types:
    head.default:
        node_config:
            instance_type:
                CPU: 4 # number of cores
                memory: 4G  # memory, for example 1024M, 1G
    worker.default:
        node_config:
            instance_type:
                CPU: 4 # number of cores
                memory: 4G # memory, for example 1024M, 1G
        min_workers: 3
        
runtimes:
    types: [hdfs, metastore, spark]
```

Execute the following command to start the cluster:
```
cloudtik start /path/to/your-cluster-config.yaml
```
