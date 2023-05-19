# Using for On-Premise Clusters
CloudTik can easily manage and scale the resources on public cloud through the cloud API/SDK.
While users sometimes want to do some performance tests on their local machines. 
In order to manage local machine resources conveniently, CloudTik has developed a Cloud Simulator service 
that runs on on-premise/private clusters to simulate cloud operations and create one or more clusters on a local machine pool.
With the cloud-simulator, CloudTik implements an on-premise provider which calls into Cloud Simulator
to create and release nodes from the machine pool.

Please follow these steps to use for On-Premise clusters .

- Prepare the machines
- Configure and start CloudTik Cloud Simulator
- Configure and start the cluster


## Prepare the machines
For the machines used for CloudTik, there are a few requirements. 
1. All the machines needs to have a user with sudo privilege. A non-root user is suggegested, for example 'cloudtik'.
2. Prepare the private key file for the clusters
3. Setup host resolution for all the nodes
4. Prepare the local disks

For public cloud providers, the virtual machines are created with #1, #2 and #3 already satisfied.
For #4 (disks) on the public cloud, CloudTik will list all raw block devices,
create a file system for them and mount to /mnt/cloudtik/data_disk_#.
CloudTik runtime will automatically search disks under /mnt/cloudtik.
And use these disks as data disks.

For on-premise clusters, user need to make sure these requirements are satisfied manually or through
utility scripts.

### Create a new sudo user
Cloudtik doesn't suggest to use root user to manage the cluster, 
so you need to create a normal user with sudo privileges for each of your machines.
If such a user already exists, you can skip this step.
For example to create cloudtik user:
```buildoutcfg
sudo useradd -ms /bin/bash -d /home/cloudtik cloudtik;
sudo usermod -aG sudo cloudtik;
echo 'cloudtik ALL=NOPASSWD: ALL' | sudo tee -a /etc/sudoer;
sudo passwd cloudtik;
```

### Prepare the private key file
We need to specify the private key file for logining to the clusters.
Use the command `ssh-keygen -t rsa -b 4096` to generate a new ssh key pair on working node.
Then add working node SSH public key to each nodes. For example: ssh-copy-id -i ~/.ssh/id_rsa.pub [sudo user]@[node-ip]


### Set up host resolution
We need to make sure that the host resolution is configured and working properly. 
This resolution can be done by using a DNS server or by configuring the "/etc/hosts" file on each node we use for cluster setting up. 
Then you also need to generate a new ssh key pair on working node and add this SSH public key to each nodes. 
Cloudtik will use this private key to login in the cluster.


### Prepare local disks
Cloudtik will automatically detect raw block devices on each node, format the disks without partitions
and mount these disks in the directory "/mnt/cloudtik/data_disk_[number]" specified by Cloudtik.

So you can either leave the data disks as raw block devices and CloudTik will do all for you
in the same way as we do for public cloud virtual machines.

For example, the disk information below, Cloudtik will use nvme0n1, nvme1n1 and nvme2n1 as storage disks.
```buildoutcfg
(base) ubuntu@worker01:~$ lsblk
NAME                      MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
loop0                       7:0    0  61.9M  1 loop /snap/core20/1494
loop1                       7:1    0  67.2M  1 loop /snap/lxd/21835
loop2                       7:2    0  61.9M  1 loop /snap/core20/1518
loop3                       7:3    0    47M  1 loop /snap/snapd/16010
loop4                       7:4    0  67.8M  1 loop /snap/lxd/22753
loop5                       7:5    0  44.7M  1 loop /snap/snapd/15904
sda                         8:0    0 372.6G  0 disk
├─sda1                      8:1    0   1.1G  0 part /boot/efi
├─sda2                      8:2    0   1.5G  0 part /boot
└─sda3                      8:3    0 370.1G  0 part
  └─ubuntu--vg-ubuntu--lv 253:0    0   100G  0 lvm  /
nvme0n1                   259:0    0   1.8T  0 disk /mnt/cloudtik/data_disk_1
nvme1n1                   259:2    0   1.8T  0 disk /mnt/cloudtik/data_disk_2
nvme2n1                   259:3    0   1.8T  0 disk /mnt/cloudtik/data_disk_3
nvme3n1                   259:5    0   3.7T  0 disk /mnt/disk_1
nvme4n1                   259:8    0   3.7T  0 disk /mnt/disk_2
nvme5n1                   259:9    0   3.7T  0 disk /mnt/disk_3
nvme6n1                   259:10   0   3.7T  0 disk /mnt/disk_4

```

Or if the data disks have already been formatted and mounted at other paths,
you can create a symbolic link from your paths to /mnt/cloudtik/data_disk_#.
By utilizing custom initialization command provided by CloudTik, you can easily to do these
in your cluster without manually doing this. For example,

```
initialization_commands:
    - sudo ln -s /mnt/your/disk_1 /mnt/cloudtik/data_disk_1
    - sudo ln -s /mnt/your/disk_2 /mnt/cloudtik/data_disk_2
```

Note: please sure your data disk paths set the right permission for read/write by all users.
If not, you can add the corresponding commands to initialization command as well.

## Configure and start CloudTik Cloud Simulator
You need prepare the CloudTik Cloud Simulator configure file and
start CloudTik Cloud Simulator service with the configure file.

### Create and configure a YAML file for cloudtik-cloud-simulator 
You need to provide your machine hardware configuration and ip.
```buildoutcfg
# Define one or more instance types with the information of its hardware resources
# Then you specify the instance type for each node in the node list
instance_types:
    head_instance_type:
        # Specify the resources of this instance type.
        resources:
            CPU: number-of-cores
            memoryMb: size-in-mb
    worker_instance_type_1:
        # Specify the resources of this instance type.
        resources:
            CPU: number-of-cores
            memoryMb: size-in-mb
    worker_instance_type_2:
        # Specify the resources of this instance type.
        resources:
            CPU: number-of-cores
            memoryMb: size-in-mb

# List of nodes with the ip and its node type defined in the above list
nodes:
    - ip: node_1_ip
      # Should be one of the instance types defined in instance_types
      instance_type: head_instance_type
      # You may need to supply a public ip for the head node if you need
      # to start cluster from outside of the cluster's network
      # external_ip: your_head_public_ip
    - ip: node_2_ip
      instance_type: worker_instance_type_1
    - ip: node_3_ip
      instance_type: worker_instance_type_1
    - ip: node_4_ip
      instance_type: worker_instance_type_2

```

### Start cloudtik-cloud-simulator service
```buildoutcfg
cloudtik-simulator [--bind-address BIND_ADDRESS] [--port PORT] your_cloudtik_simulator_config
```


## Configure and start cluster
You need prepare the cluster configure file using on-prem provider and
start the cluster with the cluster configure file.

### Create and configure a YAML file for cluster

1. On-prem provider support both docker mode and host node. 
When using docker mode and the OS of machines is RedHat-based Linux Distributions, you need to additional initialization_command to install jq.
For example,

```buildoutcfg
docker:
    # Set initialization_command to install jq if the OS is Redhat, Centos or Fedora etc. (Only need on docker mode)
    initialization_command:
	    - which jq || (sudo yum -qq update -y && sudo yum -qq install -y jq > /dev/null) 
```
2. Define cloud_simulator_address for on-prem provider. (Default port is 8282)
```buildoutcfg
# Cloud-provider specific configuration.
provider:
    type: onprem

    # We need to use Cloud Simulator for the best on-premise cluster management
    # You can launch multiple clusters on the same set of machines, and the cloud simulator
    # will assign individual nodes to clusters as needed.
    cloud_simulator_address: your-cloud-simulator-ip:port
```
3. Define ssh user and its ssh private key which are prepared above.
You also need to provide ssh_proxy_command if the head node needs to access the worker node through proxy.
```buildoutcfg
auth:
    # The ssh user configured above with sudo privilege
    ssh_user: cloudtik
    # Specify the private key file for login to the nodes
    ssh_private_key: ~/.ssh/id_rsa
    # Set proxy if you are in corporation network. For example,
    # ssh_proxy_command: "ncat --proxy-type socks5 --proxy your_proxy_host:your_proxy_port %h %p"

```
4. Define available_node_types. 
```buildoutcfg
available_node_types:
    head.default:
        node_config:
            # The instance type used here need to be defined in the instance_types
            # in the Cloud Simulator configuration file
            instance_type: head_instance_type
    worker.default:
        min_workers: 2
        node_config:
            instance_type: worker_instance_type_1
```

### Start the cluster with the configure file
Starting the cluster is simple,
```buildoutcfg
cloudtik start your_cluster_config
```
