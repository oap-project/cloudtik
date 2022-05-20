# Creating Workspace

CloudTik uses **Workspace** concept to easily manage shared Cloud resources such as VPC, network, identity resources, 
firewall or security groups. Within one workspace, you can start one or multiple clusters.

CloudTik will help users quickly create and configure:

- VPC shared by all the clusters of the workspace. 

- A private subnet for workers and a public subnet for head node. 

- Firewall rules for SSH access to head node and internal communication. 

- A NAT gateway for Internet access. 

- An identity for head node to Cloud API.


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

*NOTE:* Remember to change `CidrIp` from `0.0.0.0/0` to restricted IpRanges for TCP port 22 security as below. Replace 
`x.x.x.x/x` with your specific working node IPs.

```
    security_group:
        # Use IpPermissions to allow SSH access from your working node
        IpPermissions:
        - FromPort: 22
          ToPort: 22
          IpProtocol: TCP
          IpRanges:
          - CidrIp: x.x.x.x/x
        - FromPort: 22
          ToPort: 22
          IpProtocol: TCP
          IpRanges:
          - CidrIp: x.x.x.x/x

```

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
    # Use securityRules to allow SSH access from your working node
    securityRules:
        - properties:
            protocol: TCP
            priority: 1000
            access: Allow
            direction: Inbound
            source_address_prefixes:
              - 0.0.0.0/0
            source_port_range: "*"
            destination_address_prefix: "*"
            destination_port_range: 22

```

*NOTE:* Remember to restrict `source_address_prefixes` above to restricted range as below. Replace 
`x.x.x.x/x` with your specific working node IPs.

```
    securityRules:
        - properties:
            protocol: TCP
            priority: 1000
            access: Allow
            direction: Inbound
            source_address_prefixes:
              - x.x.x.x/x
              - x.x.x.x/x
            source_port_range: "*"
            destination_address_prefix: "*"
            destination_port_range: 22
```

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
    firewalls:
        # Use firewall_rules to allow SSH access from your working node
        # Restrict sourRanges for security
        firewall_rules:
          - allowed:
              - IPProtocol: tcp
                ports:
                  - 22
            sourceRanges:
              - 0.0.0.0/0

```

*NOTE:* Remember restrict `sourceRanges` above to restricted range according to your working node IP as below. Replace 
`x.x.x.x/x` with your specific working node IP.

```
    firewalls:
        # Use firewall_rules to allow SSH access from your working node
        # Restrict sourRanges for security
        firewall_rules:
          - allowed:
              - IPProtocol: tcp
                ports:
                  - 22
            sourceRanges:
              - x.x.x.x/x
              - x.x.x.x/x
```

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
