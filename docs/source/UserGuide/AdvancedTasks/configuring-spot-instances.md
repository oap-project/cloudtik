# Configuring Spot Instances
Cloud providers usually provide user an option to create spot instances.
Spot Instances are available at a discount compared to On-Demand prices. Cloud provider may terminate a spot instance without notification due to capacity reasons.
You can use Spot Instances for various stateless, fault-tolerant, or flexible applications such as big data, containerized workloads, CI/CD, web servers, high-performance computing (HPC), and test & development workloads.

By default, CloudTik head node will be created with non-spot instance.
And prefers spot instances for worker nodes.
You can override worker node preference on spot instances or non-spot instances.

## Disable spot instance for worker nodes
If you want to disable spot instance for worker node, you can set prefer_spot_instance to false
under the provider section. For example,

```buildoutcfg

provider:
    prefer_spot_instance: false
```

## Prefer spot instance for worker nodes
If you want to explicitly prefer spot instance for worker node, you can set prefer_spot_instance to true
under the provider section. For example,

```buildoutcfg

provider:
    prefer_spot_instance: true
```
