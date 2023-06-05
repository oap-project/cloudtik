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
CloudTik provider abstracts the hardware infrastructure layer so that CloudTik common facilities and runtimes
can consistently run on every provider environments. The support of different public cloud are implemented as providers
(such as AWS, Azure, GCP providers). Beyond the public cloud environments, we also support
virtual single node clustering, local or on-premise clusters which are also implemented as providers
(for example, virtual, local and on-premise providers)
We currently support many public cloud providers and several special providers for on-premise clusters.
More provider implementations will be added soon.
- AWS
- Azure
- GCP
- Alibaba Cloud
- Kubernetes (or EKS, AKS and GKE)
- Virtual: virtual clustering on single node
- Local: single local clustering with multiple nodes
- On-premise: cloud simulating on on-premise nodes

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
- Spark Runtime: to provide distributed analytics capabilities.
- AI Runtime: to provide distributed AI training and inference capabilities.
- Ganglia Runtime: to provide monitoring services for all nodes.
- Metastore Runtime: to provide catalog services for SQL.
- Flink Runtime: to provide distributed streaming analytics capabilities.
- Presto Runtime or Trino Runtime: to provide interactive analytics capabilities.
- Kafka Runtime: to provide event streaming services.
- Ray Runtime: to provide Ray based training and tuning capabilities.
- SSH Server Runtime: to provide password-less SSH capability within cluster.
- ZooKeeper Runtime: to provide coordinating and distributed consistency services.
