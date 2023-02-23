# Configuring Cluster Networking
From [Architecture Overview - Cluster Networking](../architecture-overview.md#cluster-networking),
we understand the cluster networking choices and how these
choices impact the VPC networking.

This section will give you the details how to configure for choosing
these network scenarios.

Basically, these choices are controlled by two options under provider section of
the workspace configuration file and cluster configuration file.
- use_internal_ips
- use_working_vpc

By default, CloudTik use VPC with public IP for head and private IPs for workers
networking model.

Note: please make sure that we use the consistent values for these options
in the workspace configuration file and the cluster configuration file
belongs to the same workspace.

## VPC with public IP for head and private IPs for workers

Set the following options in the provider section to chose
VPC with public IP for head and private IPs for workers:
```
provider:
    use_internal_ips: False
    use_working_vpc: False
```

## VPC with private IPs and VPC Peering

Set the following options in the provider section to choose
VPC with private IPs and VPC Peering.
```
provider:
    use_internal_ips: True
    use_working_vpc: False
```

## VPC with private IPs and VPC sharing

Set the following options in the provider section to choose
VPC with private IPs and VPC sharing:
```
provider:
    use_internal_ips: True
    use_working_vpc: True
```
