# Configuring User Tags
When creating VM instances, CloudTik will configure some built-in tags to the VM instances for storing
some meta information. These tags include:
- cloudtik-cluster-name
- cloudtik-node-name
- cloudtik-node-kind
- cloudtik-user-node-type
- cloudtik-node-status
- cloudtik-node-number
- cloudtik-launch-config
- cloudtik-runtime-config

User can specify additional tags if there is need.

## Head tags and worker tags
You can specify the additional tags in node_config of node type definition under available_node_types,
which provide the capability to specify tags per user node type definition.

While the format of defining tags/labels within the node config varies based on the cloud provider
as described in the next section.

## Tags configurations
This section describes the format to define tags/labels for each cloud provider.

### AWS
Use the same AWS TagSpecifications format to specify additional user tags.

For example,
```
available_node_types:   
    worker.default:
        node_config:
            TagSpecifications:
              - ResourceType: instance
                Tags:
                  - Key: my-tag
                    Value: my-tag-value
```

### GCP
Use labels field to specify a mapping of label name and value.

For example,
```
available_node_types:
    worker-default:
        node_config:
            labels:
                my-label-name: my-label-value
```

### Azure

Use tags field to specify a mapping of tag name and value.

For example,
```
available_node_types:
    worker-default:
        node_config:
            tags:
                my-tag-name: my-tag-value
```

### Kubernetes
For Kubernetes, the node config is the pod spec definition.
User can use the pod metadata labels field to specify the labels.

For example,
```
available_node_types:
    worker-default:
        node_config:
            pod:
                metadata:
                    labels:
                        my-label-name: my-label-value
```
