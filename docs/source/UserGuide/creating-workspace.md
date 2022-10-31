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
    # if you use a User account to authenticate, please add the field: gcp_credentials
    gcp_credentials:
        type: oauth_token
        credentials:
            # Option 1: You can execute 'gcloud auth print-access-token' to get oauth access token
            # and put it here as token value. You don't need to other fields for this case.
            # but note that the token is short-lived and will expire.
            token: nil
            # Option 2: Set the token above as `nil` and specify the following fields needed for getting and refreshing
            # an access token.
            # You can fill the following items with the info in 'adc.json' under '~/.config/gcloud/legacy_credentials/your_account_name'
            # directory after you authenticated your working node using gcloud with your user account.
            client_id: your-oauth-client-id
            client_secret: your-oauth-client-secret
            token_uri: https://oauth2.googleapis.com/token
            refresh_token: your-oauth-refresh-token
    # Use allowed_ssh_sources to allow SSH access from your client machine.
    # Note: For security, make sure to replace 0.0.0.0/0 with restricted CIDR ranges for your case.
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
