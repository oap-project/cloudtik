# Managing Cluster

### Status and Information

Use the following commands to show various cluster information.

```
cloudtik status /path/to/your-cluster-config.yaml
cloudtik info /path/to/your-cluster-config.yaml
cloudtik head-ip /path/to/your-cluster-config.yaml
cloudtik worker-ips /path/to/your-cluster-config.yaml
cloudtik process-status /path/to/your-cluster-config.yaml
cloudtik monitor /path/to/your-cluster-config.yaml
```

### Attach to Cluster Nodes

```
cloudtik attach /path/to/your-cluster-config.yaml
``` 

### Execute and Submit Jobs

```
cloudtik exec /path/to/your-cluster-config.yaml
```

### Managing Files

###### Copy local files to cluster head (or to all nodes)

```
cloudtik rsync-up /path/to/your-cluster-config.yaml [source] [target]
```

###### Copy file from cluster to local
```
cloudtik rsync-down /path/to/your-cluster-config.yaml [source] [target]
```

### Start or Stop Runtime Services

### Scale Up or Scale Down Cluster

### Accessing the Web UI



For more information as to the commands, you can use `cloudtik --help` or `cloudtik [command] --help` to get detailed instructions.