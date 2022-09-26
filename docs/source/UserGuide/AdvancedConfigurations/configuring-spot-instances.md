# Configuring Spot Instances
Cloud providers usually provide user an option to create spot instances.
Spot Instances are available at a discount compared to On-Demand prices. Cloud provider may terminate a spot instance without notification due to capacity reasons.
You can use Spot Instances for various stateless, fault-tolerant, or flexible applications such as big data, containerized workloads, CI/CD, web servers, high-performance computing (HPC), and test & development workloads.

By default, CloudTik head node and worker nodes will be created with non-spot instance.
You can override worker node preference on spot instances or non-spot instances.
Please remember that if you set to prefer spot nodes, you may face failures on creating
workers for a specific type if that instance type of spot instances are not available.
You can switch to non-spot instances if this happens.

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
