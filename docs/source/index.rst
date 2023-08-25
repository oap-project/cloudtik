CloudTik
========

CloudTik is a cloud scale platform for distributed analytics and AI on public cloud providers including AWS, Azure, GCP, and so on.
CloudTik enables users or enterprises to easily create and manage analytics and AI platform on public clouds,
with out-of-box optimized functionalities and performance, and to go quickly to focus on running the business workloads
in hours or in even minutes instead of spending months to construct and optimize the platform.

CloudTik is designed as a platform to help user focuses on business development and
achieve "Develop once, run everywhere" with the following core capabilities:

* Scalable, robust, and unified control plane and runtimes for all environments:

  * Public cloud providers and Kubernetes
  * Single node virtual clustering
  * Local or on-premise clusters
* Out of box optimized runtimes for analytics and AI

  * Optimized Spark runtime with CloudTik optimizations
  * Optimized AI runtime with Intel oneAPI
* Infrastructure and runtimes to support microservices orchestration with:

  * Service discovery - service registry, service discover, service DNS naming
  * Load balancing -  Layer 4 or Layer 7 load balancer working with built-in service discovery
* Support of major public cloud providers:

  * AWS - Amazon Elastic Compute Cloud (EC2) or Amazon Elastic Kubernetes Service (EKS)
  * Azure - Azure Virtual Machines or Azure Kubernetes Service (AKS)
  * GCP -  Google Compute Engine (GCE) or Google Kubernetes Engine (GKE)
  * Alibaba Cloud - Elastic Compute Service (ECS)
  * Kubernetes and more
* A fully open architecture and open-sourced platform

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   GettingStarted/introduction.md
   GettingStarted/quick-start.md

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   UserGuide/key-concepts.md
   UserGuide/architecture-overview.md
   UserGuide/installation.md
   UserGuide/configuring-proxy-and-firewall.md
   UserGuide/login-to-cloud.md
   UserGuide/configuring-cloud-storage.md
   UserGuide/configuring-cloud-database.md
   UserGuide/creating-workspace.md
   UserGuide/creating-cluster.md
   UserGuide/managing-cluster.md
   UserGuide/configuring-kubernetes-integration.md
   UserGuide/running-locally.rst
   UserGuide/running-optimized-analytics.rst
   UserGuide/running-optimized-ai.rst
   UserGuide/advanced-configuring.rst


.. toctree::
   :maxdepth: 1
   :caption: Reference

   Reference/configuration.rst
   Reference/command-reference.md
   Reference/providers.md
   Reference/runtimes.md
   Reference/api-reference.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   DeveloperGuide/building-cloudtik.md
   DeveloperGuide/developer-api.md
