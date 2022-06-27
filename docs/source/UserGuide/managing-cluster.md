# Managing Cluster

After a cluster is created, you can use CloudTik to manage the cluster and submit jobs.

### Status and Information

Use the following commands to show various cluster information.

```
# Check cluster status with:
cloudtik status /path/to/your-cluster-config.yaml
# Show cluster summary information and useful links to connect to cluster web UI.
cloudtik info /path/to/your-cluster-config.yaml
cloudtik head-ip /path/to/your-cluster-config.yaml
cloudtik worker-ips /path/to/your-cluster-config.yaml
cloudtik process-status /path/to/your-cluster-config.yaml
cloudtik monitor /path/to/your-cluster-config.yaml
cloudtik debug-status /path/to/your-cluster-config.yaml
cloudtik health-check  /path/to/your-cluster-config.yaml
```

Here are examples to execute these CloudTik CLI commands on GCP clusters.

```
$ cloudtik status /path/to/your-cluster-config.yaml
 
Total 2 nodes. 2 nodes are ready
+---------------------+----------+-----------+-------------+---------------+---------------+-----------------+
|       node-id       | node-ip  | node-type | node-status | instance-type |   public-ip   | instance-status |
+---------------------+----------+-----------+-------------+---------------+---------------+-----------------+
| 491xxxxxxxxxxxxxxxxx| 10.0.x.x |    head   |  up-to-date | n2-standard-4 | 23.xxx.xx.xxx |     RUNNING     |
| 812xxxxxxxxxxxxxxxx | 10.0.x.x |   worker  |  up-to-date | n2-standard-4 |      None     |     RUNNING     |
+---------------------+----------+-----------+-------------+---------------+---------------+-----------------+
```

Show cluster summary information and useful links to connect to cluster web UI.

```
$ cloudtik info /path/to/your-cluster-config.yaml
 
Cluster small is: RUNNING
1 worker(s) are running

Runtimes: ganglia, spark

The total worker CPUs: 4.
The total worker memory: 16.0GB.

Key information:
    Cluster private key file: *.pem
    Please keep the cluster private key file safe.

Useful commands:
  Check cluster status with:
...

```

Show debug status of cluster scaling.

```
$ cloudtik debug-status /path/to/your-cluster-config.yaml

======== Cluster Scaler status: 2022-05-12 08:46:49.897707 ========
Node status
-------------------------------------------------------------------
Healthy:
 1 head-default
 1 worker-default
Pending:
 (no pending nodes)
Recent failures:
 (no failures)
```

Check if this cluster is healthy.

```
$ cloudtik health-check  /path/to/your-cluster-config.yaml
Cluster is healthy.
```

### Attach to Cluster Nodes

Connect to a terminal of cluster head node.

```
cloudtik attach /path/to/your-cluster-config.yaml
```

Then you will log in to head node of the cluster vis SSH as below

```
$ cloudtik attach /path/to/your-cluster-config.yaml
(base) ubuntu@cloudtik-example-head-a7xxxxxx-compute:~$
```

Log in to a worker node with `--node-ip` as below.

```
$ cloudtik attach --node-ip 10.0.x.x /path/to/your-cluster-config.yaml
(base) ubuntu@cloudtik-example-worker-150xxxxx-compute:~$
```

### Execute and Submit Jobs

Execute a command via SSH on head node.

```
cloudtik exec /path/to/your-cluster-config.yaml [CMD]
```

For example, list the items under $USER directory as below.  

```
$ cloudtik exec /path/to/your-cluster-config.yaml ls
anaconda3  cloudtik_bootstrap_config.yaml  cloudtik_bootstrap_key.pem  jupyter  runtime
```

Execute commands on specified worker node 

```
cloudtik exec --node-ip x.x.x.x /path/to/your-cluster-config.yaml [CMD]
```

Execute commands on all nodes

```
cloudtik exec --all-nodes /path/to/your-cluster-config.yaml [CMD]
```

### Manage Files

Upload files or directories to cluster:

``` 
cloudtik rsync-up /path/to/your-cluster-config.yaml [source] [target]
```
  
Download files or directories from cluster

```
cloudtik rsync-down /path/to/your-cluster-config.yaml [source] [target]
```

### Start or Stop Runtime Services

```
cloudtik runtime start /path/to/your-cluster-config.yaml
cloudtik runtime stop /path/to/your-cluster-config.yaml
```

### Scale Up or Scale Down Cluster

Scale up the cluster with a specific number cpus or nodes.

Scale up the cluster by specifying `--cpus` as below.

```
$ cloudtik scale  --cpus 12 /path/to/your-cluster-config.yaml
Are you sure that you want to scale cluster small to 12 CPUs? Confirm [y/N]: y

Shared connection to 23.xxx.xx.xxx closed.

$ cloudtik info /path/to/your-cluster-config.yaml
Cluster small is: RUNNING
2 worker(s) are running

Runtimes: ganglia, spark

The total worker CPUs: 8.
The total worker memory: 32.0GB.
...

```

Scale up the cluster by specifying `--nodes` as below.

```
$ cloudtik scale --nodes 4  /path/to/your-cluster-config.yaml
Are you sure that you want to scale cluster small to 4 nodes? Confirm [y/N]: y

Shared connection to 23.xxx.xx.xxx closed.
$ cloudtik info /path/to/your-cluster-config.yaml
Cluster small is: RUNNING
3 worker(s) are running

Runtimes: ganglia, spark

The total worker CPUs: 12.
The total worker memory: 48.0GB.

Key information:
...

```

### Access the Web UI

```
The SOCKS5 proxy to access the cluster Web UI from local browsers:
    localhost:6001

Ganglia Web UI:
    http://<head-internal-ip>/ganglia
YARN ResourceManager Web UI:
    http://<head-internal-ip>:8088
Spark History Server Web UI:
    http://<head-internal-ip>10.0.0.4:18080
Jupyter Web UI:
    http://<head-internal-ip>:8888, default password is 'cloudtik'
```


For more information as to the commands, you can use `cloudtik --help` or `cloudtik [command] --help` to get detailed instructions.