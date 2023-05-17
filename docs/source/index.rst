CloudTik
========

CloudTik is a cloud scale platform for distributed analytics and AI on public cloud providers including AWS, Azure, GCP, and so on.
CloudTik enables users or enterprises to easily create and manage analytics and AI platform on public clouds,
with out-of-box optimized functionalities and performance, and to go quickly to focus on running the business workloads
in hours or in even minutes instead of spending months to construct and optimize the platform.

CloudTik provides:

* Scalable, robust, and unified control plane and runtimes for all public clouds
* Out of box optimized runtimes for analytics and AI

  * Optimized Spark runtime with CloudTik optimizations
  * Optimized AI runtime with Intel oneAPI
* Support of major public cloud providers - AWS, Azure, GCP, Kubernetes (EKS, AKS, and GKE) and more
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
   UserGuide/creating-workspace.md
   UserGuide/creating-cluster.md
   UserGuide/managing-cluster.md
   UserGuide/configuring-kubernetes-integration.md
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
