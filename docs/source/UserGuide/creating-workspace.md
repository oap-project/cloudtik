# Creating Workspace

Creating a workspace is simple. The first step is to create a workspace configuration file.
The key information in the workspace configuration file usually includes:
- A unique workspace name
- A few provider specific key properties such as provider type, region/location
- One or more allowed ssh sources which allowing your working machine SSH to cloud

## Creating a Workspace Configuration File

A typical workspace configuration file is simple. Specify the unique workspace name, cloud provider type
and a few provider-specific properties. 

### AWS

Here is an AWS workspace configuration yaml example, which is located at CloudTik's `example/cluster/aws/example-workspace.yaml` 

```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    # Use allowed_ssh_sources to allow SSH access from your client machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```

*NOTE:* `0.0.0.0/0` in `allowed_ssh_sources` will allow any IP addresses to connect to your cluster as long as it has the cluster private key.
For more security, make sure to change from `0.0.0.0/0` to restricted CIDR ranges for your case.


### Azure

Here is an Azure workspace configuration yaml example, which is located at CloudTik's `example/cluster/azure/example-workspace.yaml`

```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: azure
    location: westus
    subscription_id: your_subscription_id
    # Use allowed_ssh_sources to allow SSH access from your client machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```

*NOTE:* `0.0.0.0/0` in `allowed_ssh_sources` will allow any IP addresses to connect to your cluster as long as it has the cluster private key.
For more security, make sure to change from `0.0.0.0/0` to restricted CIDR ranges for your case.

### GCP

```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: gcp
    region: us-central1
    availability_zone: us-central1-a
    project_id: your_project_id
    # Use allowed_ssh_sources to allow SSH access from your client machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```

*NOTE:* `0.0.0.0/0` in `allowed_ssh_sources` will allow any IP addresses to connect to your cluster as long as it has the cluster private key.
For more security, make sure to change from `0.0.0.0/0` to restricted CIDR ranges for your case.

## Creating or Deleting a Workspace

Use the following command to create and provision a workspace:

```
cloudtik workspace create /path/to/<your-workspace-config>.yaml
```

After the workspace is created, shared cloud resources such as VPC, network, identity resources, firewall or security 
groups are configured.

Use the following command to delete a workspace:

```
cloudtik workspace delete /path/to/<your-workspace-config>.yaml
```

Check `./example/cluster` folder for more Workspace configuration file examples.
