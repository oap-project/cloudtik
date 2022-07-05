# Switching Client Machine
Sometimes, you may need to work on another client machine (working machine).
This guide will introduce the key aspects when you created a cluster from one client machine,
and you need to switch to another client machine to operate the cluster through CloudTik CLI. 

## Moving the configuration files 
You need move the cluster configuration file to the new client machine
so that you can issue CloudTik CLI command to the cluster specifying the same cluster configuration file.

If you want to manage workspace on the new client machine, backup and move the workspace configuration file as well.

## Moving the cluster private key file
The cluster was accessed using the cluster private key file.
If you don't know where is the private key file located, you can execute:

```bash
   cloudtik info your-cluster-config.yaml
```
The command will show where is the cluster private key located.

You need moving the corresponding private key file and for azure also the public key file to the new client machine.
The file names of the private or public key may have association with the key pair name at Cloud side (for AWS and GCP).
Please don't change the file names of the private or public key files when you are doing file movements.

Note: When you are moving key files, please make sure to set the private key file attributes to read/write by the owner only (0600 rw-------)
at target machine. Otherwise, SSH will not recognize your private key file.
 
For a more detailed description of cluster key file, refer to [Understanding Cluster Key](./understanding-cluster-key.md)