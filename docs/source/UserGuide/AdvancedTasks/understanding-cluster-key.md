# Understanding Cluster Key
The cluster is protected by the cluster key file (private key file and optionally public key file).

The public key is used to when creating VM instance. For different Cloud providers, the way of specifying
public key when creating VM instance is different.
For AWS, AWS key pair is used to associate the name with the public and private key
and the key pair name is specified when creating VM instance.
For Azure, the public key is specified directly when creating VM instance.
For GCP, we can create project wide ssh key to associate a USERNAME with its public key.
And all VM instances in the same project can be accessed using the private key of the project wide
ssh key.

For private key, it is simpler. CloudTik CLI needs the cluster private key file to connect to the cluster
and issue management commands for all Cloud providers.

Note: If you are creating SSH key pair manually, please make sure that the private key file attributes are read/write by the owner only (0600 rw-------).

## The location of cluster key files
For a running cluster, if you don't know where is the cluster key files located,
you can execute info command,

```bash
   cloudtik info your-cluster-config.yaml
```
The command will show where is the cluster private or public key files located.

## AWS cluster key

### Implicit cluster private key
If you don't specify a private key through ssh_private_key in the auth section in cluster configuration file,
CloudTik will try to find an existing AWS key pair with the name pattern of cloudtik_aws_{region}_{index}
and if the private key file of the key pair ~/.ssh/cloudtik_aws_{region}_{index}.pem exists.
If found, the cluster will be created with the key pair and the private key file at ~/.ssh/cloudtik_aws_{region}_{index}.pem.

Otherwise, CloudTik will create a new key pair with the name pattern of cloudtik_aws_{region}_{index}
and download the private key file of the key pair to ~/.ssh/cloudtik_aws_{region}_{index}.pem.

```buildoutcfg
auth:
    ssh_user: ubuntu
```

If you create multiple clusters on the same client machine, CloudTik will use the same cluster private key based
on the above process. If you don't want this, you can specify explicitly a key pair name to use for the cluster
with provider/key_pair/key_name configuration key in cluster configuration file.

```buildoutcfg
provider:
    type: aws
    key_pair:
        key_name: my_cluster_key_name
auth:
    ssh_user: ubuntu
```

### Explicit cluster private key
You can also specify explicitly a private key file in the auth section in cluster configuration file.
In this case, the specified private key file will be used.
And you also need to explicitly specify the AWS key pair name of the private key through 'KeyName' configuration key
in the 'node_config' of each node types defined in 'available_node_types'.

```buildoutcfg

available_node_types
    head.default:
        node_config:
            KeyName: your_aws_key_pair_name
    worker.default:
        node_config:
            KeyName: your_aws_key_pair_name
auth:
    ssh_user: ubuntu
    ssh_private_key: ~/.ssh/your_aws_ssh_private_key.pem
```

For this case, CloudTik will not try creating any key pairs on AWS.
You need to make sure the key pair exists on AWS and the private key file contains the corresponding private key material.


## Azure cluster private key
Azure cluster key management is simple.
When creating the cluster, you need to generate a cluster RSA key pair.
And specify the ssh_private_key configuration key and ssh_public_key configuration key
in auth section for locating the private and public key file.

The public key file will be used to create VM instances and the private key file
can be used to access the VM instances through SSH.

## GCP cluster private key

### Implicit cluster private key
If you don't specify a private key through ssh_private_key in the auth section in cluster configuration file,
CloudTik will try to find an existing GCP project wide ssh key with the USERNAME equals to ssh_user in the configuration file
and if the private key file and public key file of the key pair ~/.ssh/cloudtik_gcp_{region}_{project_id}_{ssh_user}_{index}.pem
and ~/.ssh/cloudtik_gcp_{region}_{project_id}_{ssh_user}_{index}.pub exist locally.
If found, the cluster will be created with the ssh key and using the private key file at ~/.ssh/cloudtik_gcp_{region}_{project_id}_{ssh_user}_{index}.pem
for cluster access.

Otherwise, CloudTik will generate an RSA key pair
and create a GCP project wide ssh key with the USERNAME equals to ssh_user and the public key of the generated RSA key pair.
and save the private key file of the key pair to ~/.ssh/cloudtik_gcp_{region}_{project_id}_{ssh_user}_{index}.pem and
public key file of the key pair to ~/.ssh/cloudtik_gcp_{region}_{project_id}_{ssh_user}_{index}.pub.

```buildoutcfg
auth:
    ssh_user: ubuntu
```

If you create multiple clusters on the same client machine and used the same ssh_user, CloudTik will use the same cluster private key based
on the above process.

### Explicit cluster private key
You can also specify explicitly a private key file in the auth section in cluster configuration file.
In this case, the specified private key file will be used.

For this case, CloudTik will not try creating any project wide ssh key on GCP.
You need to make sure the project wide ssh key exists with USERNAME equals to ssh_user exists on GCP project and the private key file contains the corresponding private key material.

```buildoutcfg
auth:
    ssh_user: ubuntu
    ssh_private_key: ~/.ssh/your_gcp_ssh_private_key.pem
```
