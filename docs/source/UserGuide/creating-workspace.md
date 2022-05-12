# Creating Workspace

CloudTik uses **Workspace** concept to manage your Cloud network and other resources. Within one workspace, you can start one or multiple clusters.

Use the following command to create and provision a workspace:

```
cloudtik workspace create /path/to/<your-workspace-config>.yaml
```

A typical workspace configuration file is usually very simple. Specify the unique workspace name, cloud provider type
and a few provider-specific properties. 

Here take AWS for example.

```
# A unique identifier for the workspace.
workspace_name: example-workspace

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    security_group:
        # Use IpPermissions to allow SSH access from your working node
        IpPermissions:
        - FromPort: 22
          ToPort: 22
          IpProtocol: TCP
          # restrict IpRanges here according to your cluster for security
          IpRanges:
          - CidrIp: 0.0.0.0/0
```

Check `./example/cluster` folder for more Workspace configuration file examples.