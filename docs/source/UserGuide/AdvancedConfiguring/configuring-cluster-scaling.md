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

CloudTik built-in with 3 auto-scaling policy for use:
- Scaling with Load
- Scaling with Time
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


### Scaling with Time
If you want to scale the cluster based on the time of a day
use this scaling policy.

To use this scaling policy, specify the following runtime configuration.
For example,

```
runtime:
    scaling:
        scaling_policy: scaling-with-time
        scaling_periodic: daily
        scaling_math_base: on-min-workers
        scaling_time_table:
            "8:00": "+1"
            "9:00": "*2"
            "10:00": "*0.5"
            "11:00": "-1"
```

This will enable and use scaling-with-time policy.

- scaling_periodic: The periodic interval for a scaling cycle.
  - daily: Each day will be a cycle. For daily cycle, the time format in the time table is HH:MM:SS. Minutes and seconds are optional.
  - weekly: Each week will be a cycle. For weekly cycle, the time format in the time table is, for example "Tue 10:00:00"
  - monthly: Each month will be a cycle. For monthly cycle, the time format in the time table is, for example "20 10:00:00"
- scaling_math_base: The base nodes used to do math such as *n or +n or -n.
  - on-min-workers: the min_workers of cluster will be used. if min workers is 3, "*3" will get 9 nodes.
  - on-previous-time: the nodes of previous time is used, if previous time is 2 nodes, "*3" will 6 nodes.
    Please note that when this type is used, at least item in the time table must be a specific node number.
- scaling_time_table: The time table for nodes to scale. The value can be:
  - A specific node number. Use 0 to refer to the min workers.
  - A multiplier, addition or reduction on a base. For example, "*2.5", "*3", "+4", "-5"

#### scaling_math_base examples

The following example uses on-min-workers for scaling_math_base option:
```
runtime:
    scaling:
        scaling_policy: scaling-with-time
        scaling_periodic: daily
        scaling_math_base: on-min-workers
        scaling_time_table:
            "8:00": "+1"
            "9:00": "+2"
            "10:00": "*2"
            "11:00": "-1"
            "15:00": "*3"
            "16:00": "+1"
```

will scale the cluster as following, if min_workers is 3:

```
            "8:00": 4 nodes
            "9:00": 5 nodes
            "10:00": 6 nodes
            "11:00": 2 nodes
            "15:00": 9 nodes
            "16:00": 4 nodes
```

The following example uses on-previous-time for scaling_math_base option:
```
runtime:
    scaling:
        scaling_policy: scaling-with-time
        scaling_periodic: daily
        scaling_math_base: on-previous-time
        scaling_time_table:
            "7:00": 5
            "8:00": "+1"
            "9:00": "+2"
            "10:00": "*2"
            "11:00": "-6"
            "15:00": "*0.5"
```

The above time table will resolve to the following scaling schedule:
```
            "7:00": 5 nodes
            "8:00": 6 nodes
            "9:00": 8 nodes
            "10:00": 16 nodes
            "11:00": 10 nodes
            "15:00": 5 nodes
```

#### scaling_periodic examples
The following example uses weekly as cycle which on Monday 7:00, scale up
double the size of the cluster and scale down to normal on weekends.

```
runtime:
    scaling:
        scaling_policy: scaling-with-time
        scaling_periodic: weekly
        scaling_math_base: on-min-workers
        scaling_time_table:
            "Mon 07:00": "*2"
            "Fri 19:00": "*1"
```

The following example uses monthly as cycle which on 10 7:00 every month, scale up
double the size of the cluster and scale down to normal on 20 19:00.

```
runtime:
    scaling:
        scaling_policy: scaling-with-time
        scaling_periodic: weekly
        scaling_math_base: on-min-workers
        scaling_time_table:
            "10 07:00": "*2"
            "20 19:00": "*1"
```


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

### Scaling down
When an auto-scaling policy is used and the cluster is scaled up based on
the configured conditions, the auto-scaling down will happen if there are no more
resource requests. Any nodes in idle status will be terminated to reach
the minimum number of workers or manually scaled number of workers.

The auto-scaling policy will override the way of deciding a worker's idle status.
For example, for Scaling with Load, the CPU utilization is used to decide whether
a node is idle or not; for Scaling with Spark, if there is no YARN container running
on a node, the worker is considered to be idle.

## Configuring idle time for node termination
For either Manual scaling or Auto scaling, CloudTik detects worker idle state
for removing to scale down.

Just described above the scaling policy and CloudTik default provide
ways to decide whether a node is in use (not idle) or not by CPU utilization
or runtime metrics.

While it doesn't mean a node is in idle state for one or two seconds
will be removed immediately. For robust, we used an idle timeout setting
which is default to 5 minutes which means a worker will be the candidate
to be removed only if the worker is idle in at least last 5 minutes.

User can override the value in cluster configuration.
For example, you in cluster configuration, add following to override
the idle timeout to 10 minutes.

```
# If a node is idle for this many minutes, it will be removed.
idle_timeout_minutes: 10
```
