# Using Templates
For cluster configurations, user usually needs the highest flexibility on configurations for
addressing different requirements and use cases. While on the other hand, user want a system to
be most convenient and be as simple as possible even for complex requirements.

CloudTik templates is for this purpose and designed to make the configurations as simple and reusable
for different cloud providers.

## What's template
For cluster configurations, most complex variations usually come from the instance configurations for
a specific cloud provider. For example, on AWS, for some cases, we need m5.4xlarge instance worker
with 2 additional SSD attached on AWS. For some other cases, we need m5.16xlarge instance worker
with 4 additional SSD attached. These requirements may happen in a very similar way for other cloud providers
like Azure or GCP. Configuring the instance and additional disks are not simple for many users.

CloudTik designs a templating structure to allow user to reuse typical configurations. And we also provide
built-in templates for user to use directly for typical instance configurations optimized for Spark.

A template is fundamentally a cluster configuration file which can be used and inherited by 'from' statement.

For example, below is a built-in template named 'standard' for AWS which defines instance types and disk configurations
for both head and worker node,

```
# Cloud-provider specific configuration.
provider:
    type: aws

# The instance configuration for a standard instance type
available_node_types:
    head.default:
        node_config:
            InstanceType: m5.2xlarge
            BlockDeviceMappings:
                - DeviceName: /dev/sda1
                  Ebs:
                      VolumeSize: 100
                      VolumeType: gp2
                      DeleteOnTermination: True
    worker.default:
        node_config:
            InstanceType: m5.2xlarge
            BlockDeviceMappings:
                - DeviceName: /dev/sda1
                  Ebs:
                      VolumeSize: 100
                      VolumeType: gp2
                      DeleteOnTermination: True
                - DeviceName: /dev/sdf
                  Ebs:
                      VolumeSize: 200
                      VolumeType: gp3
                      # gp3: 3,000-16,000 IOPS
                      Iops: 5000
                      DeleteOnTermination: True
```

## Using built-in templates
Using built-in templates is simple. You can simply use a 'from' statement in your
cluster configuration file and specify a built-in template name.

For example, to use the above 'standard' template for AWS,

```
# An example of standard 1 + 3 nodes cluster with standard instance type
from: aws/standard

# A unique identifier for the cluster.
cluster_name: example-standard

# Workspace into which to launch the cluster
workspace_name: exmaple-workspace

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2

available_node_types:
    worker.default:
        # The minimum number of worker nodes to launch.
        min_workers: 3
```

## Built-in templates
CloudTik defines a lot of built-in templates optimized for Spark.
These templates can be found in 'templates' folder under your CloudTik python package.
The template name used in cluster configuration is relative to templates folder.
In the folder, we defined various templates for cloud providers such as AWS, Azure and GCP,
which use can use directly like above example.

```
templates
    aws
        small
        standard
        medium
        large
        very-large
    azure
        small
        standard
        medium
        large
        very-large
    gcp
        small
        standard
        medium
        large
        very-large
```

For each template, we defined a high-memory version which uses the memory optimized
instance of the corresponding provider. For example, for 'aws/standard', 'aws/standard-highmem' is available.

For each template, we also defined the latest version which uses the latest generation
of the instance from the provider.
For example, for 'aws/standard', we defined 'aws/latest/standard' which uses 'm6i.2xlarge' instance.

## Defining your own template
A template can be used (or inherited) using 'from' statement.
You can define your own template from beginning or inheriting from an existing template.

For example, following defines a new template from aws/standard and replacing
the worker instance type with m5.4xlarge.

```
from: aws/standard

available_node_types:
    worker.default:
        node_config:
            InstanceType: m5.4xlarge
```

## Using your own template
Built-in template can be used simply by 'from' statement. User defined template are used in
the same way but need one additional configuration to allow the system to found it.

User needs to set an OS environment variable 'CLOUDTIK_USER_TEMPLATES' with the full path of
the location which contains the template sub folders and files.

For example,
```

home
    cloudtik
        my-templates
            aws
                my-template-1
            azure
                my-template-2
            production
                my-template-3
```
And you can set CLOUDTIK_USER_TEMPLATES as following,
```
export CLOUDTIK_USER_TEMPLATES=/home/cloudtik/my-templates
```

'CLOUDTIK_USER_TEMPLATES' can be a comma separated paths for multiple locations.
The template name is the relative path to the root folder, which allows user to specify
a folder with sub-folders which in turn contains the template file just like built-in templates.

For example, to use,

```
from: production/my-template-3
```
