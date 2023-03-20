# Configuring Cluster Scaling
CloudTik provide 3 mechanisms to decide the cluster size: 
- Initial size: User can get an initial size of cluster by setting "min_workers" in the node type config.
- Manual scaling: User can scale up or scale down to specific size by executing "cloudtik scale" commands.
- Auto scaling: User can configure an auto scaling policy to dynamically scale up.

## Initial size
After the cluster head node is bootstrapped,
the cluster controller on head will scale the cluster up the minimum number of workers.

The minimum number of worker nodes is configured per node type
in cluster configuration, and the default number is 1. User can override for example,

On AWS, Azure and other:
```
available_node_types:
    worker.default:
        min_workers: 8
```

On GCP:
```
available_node_types:
    worker-default:
        min_workers: 8
```

Cluster controller will monitor the health of the nodes and will make sure there are
minimum number of healthy workers. For example, one worker gets unhealthy,
and CloudTik will try first to recovery it and if cannot, kill it and start a new one.

Note that the cluster will not scale down below to the minimum number of workers
even the workers are idle. 

## Manual scaling
When the initial size of the cluster is not enough, user can manually scale up the cluster
by using "cloudtik scale" command.

User specify the number workers of the cluster or the number of the total cores of the cluster.

```
cloudtik scale /path/to/your-cluster-config.yaml --workers=8
```

And if you need to scale down, execute the scale command with smaller number.

```
cloudtik scale /path/to/your-cluster-config.yaml --workers=3
```

The cluster controller will terminate the idle workers to scale down.
Please note that the workers may not be terminated immediately if they are not idle. 

By default, the cluster controller will use the CPU utilization to decide whether
a worker is idle or not.
If there is auto-scaling policy in use, the auto-scaling policy will override the way
of deciding a worker's idle status.

## Auto scaling
If user want to automatically scaling up or down based on system metrics such as
the system load, user can use auto-scaling.

CloudTik built-in with two auto-scaling policy for use:
- Scaling with Load
- Scaling with Spark

### Scaling with Load
If you want to scale the cluster based on the CPU or memory utilization (load),
use this scaling policy.

To use this scaling policy, specify the following runtime configuration,

```
runtime:
    scaling:
        scaling_policy: scaling-with-load
```

This will enable and use scaling-with-load policy with default parameters.
User can override the parameter values,

```
runtime:
    scaling:
        scaling_policy: scaling-with-load
        scaling_resource: CPU
        scaling_step: 2
        cpu_load_threshold: 0.85
        memory_load_threshold: 0.85
        in_use_cpu_load_threshold: 0.15
```
- scaling_policy: The scaling policy name: scaling-with-load or none
- scaling_resource: The resource type to check for scale: CPU or memory
- scaling_step: The number of nodes for each scale up step
- cpu_load_threshold: The cpu load threshold above which to start scale
- memory_load_threshold: The memory load threshold above which to start scale
- in_use_cpu_load_threshold: The minimum cpu load to consider the machine is in use


### Scaling with Spark
If you want to scale the cluster based Spark application and resource utilization
tracked by YARN, use this scaling policy.

To use this scaling policy, specify the following configuration in spark runtime
configuration,

```
runtime:
    spark:
        scaling:
            scaling_mode: apps-pending
```

This will enable and use Spark scaling policy based on apps-pending with default parameters.
User can override the parameter values,

```
runtime:
    spark:
        scaling:
            scaling_mode: apps-pending
            scaling_resource: CPU
            scaling_step: 2
            apps_pending_threshold: 0.85
            apps_pending_free_cores_threshold: 4
            apps_pending_free_memory_threshold: 1024
            aggressive_free_ratio_threshold: 0.1
```
- scaling_mode: The Spark scaling mode. Values: apps-pending, aggressive or none
- scaling_resource: The resource type to check for scale: CPU or memory
- scaling_step: The number of nodes for each scale up step
- apps_pending_threshold: The number of pending apps threshold above which to trigger scaling
- apps_pending_free_cores_threshold: The number of free cores threshold below which to trigger scaling
- apps_pending_free_memory_threshold: The size of free memory threshold in MB below which to trigger scaling
- aggressive_free_ratio_threshold: The free cores or memory ratio below which to trigger scaling for aggressive mode
