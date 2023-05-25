# Key Concepts

This section gives an introduction to CloudTik’s key concepts.

- [Workspace](#workspace)
- [Cluster](#cluster)
- [Provider](#provider)
- [Runtime](#runtime)

## Workspace

CloudTik uses **Workspace** concept to manage shared Cloud specific resources which needed for running a production level
platform. These resources include
- VPC and subnets
- firewall or security groups
- gateways
- identities and roles

These resources are usually created once and shared among different clusters. CloudTik provides the commands to help user
create or delete a workspace on a specific Cloud, which will provision and configured the right resources to be ready to use.

**Note: Some resources like NAT gateway or elastic IP resources in Workspace cost money.
The price policy may vary among cloud providers.
Please check the price policy of the specific cloud provider to avoid undesired cost.**

User can create one or more workspaces on a Cloud provider as long as the resources limits are not reached.
Since some resources in Workspace cost money on time basis,
to save your cost, don't create unnecessary workspaces.

Within each workspace, user can start one or more clusters with necessary runtime components provisioned and configured in use.


## Cluster

Cluster is a scalable collection of nodes with necessary runtime services running and provides analytics and AI services
to the user. A Cluster is a self-managed organism which is not only include the hardware infrastructure but also the services
that running over it and works together to form a self-recoverable and servicing organism.

A CloudTik cluster consists of one head node and zero or more worker nodes. The head node is the brain of the cluster
and runs services for creating, setting up and starting new worker nodes, recovering an unhealthy worker node and so on.
Cloudtik provides powerful and easy to use facilities for users to quickly create and manage analytics and AI clusters.

A cluster belongs to a Workspace and shares the common workspace resources including VPC, subnets, gateways and so on.
All clusters within the same workspace are network connected and are accessible by each other.

## Provider

In the high level, provider refers a public cloud provider which provide the infrastructure as a service
for CloudTik to run on. We currently support many public cloud providers and several special providers for on-premise clusters.
More provider implementations will be added soon.
- AWS
- Azure
- GCP
- Alibaba Cloud
- On-premise: cloud simulating on on-premise nodes
- Local: local clustering with multiple nodes
- Virtual: virtual clustering with docker containers on a single node


In the design level, provider is a concept to abstract out the difference of public providers. With this abstraction, CloudTik is designed to share
the same user experiences among different cloud providers as much as possible. Internally, we have two aspects of abstraction
for a specific provider:
- Node Provider: Abstraction of how a cloud provider to provide services of creating or terminating instances and instances managing functionalities.
- Workspace Provider: Abstraction of how a cloud provider to implement the CloudTik workspace design on network and securities.

## Runtime

Cloudtik introduces **Runtime** concept to abstract a distributed service or a collection of related services running
on head and worker nodes. Different runtimes share some high level but common patterns like installing, configuring,
starting, stopping and so on. CloudTik provides infrastructure for orchestration of these aspects and allow a runtime
to handle its very specific implementation. In this way, a runtime can easy be implemented and orchestrated into
CloudTik and works together with other runtimes by providing services to other runtimes or consuming other runtime services.

Currently, we implemented the following runtimes:
- Ganglia: To provide monitoring services for all nodes.
- Metastore: To provide catalog services for SQL.
- Spark: To provide analytics capabilities.
- Presto: To provide interactive analytics capabilities.
- ZooKeeper: To provide coordinating and distributed consistency services.
- Kafka: To provide event streaming services.
