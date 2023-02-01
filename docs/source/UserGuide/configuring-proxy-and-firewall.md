# Configuring Proxy and Firewall
Proxy and firewall settings are very critical for network connectivity operating cloud instances
using CloudTik from your working machine.

The proxy settings is crucial for your working machine in a private network to be able to connect to public internet
where the cloud instance (CloudTik head node usually have a public IP) is running.

The firewall settings are the cloud firewall settings to unblock your working machine at the cloud side
so that the working machine is allowed to access the cloud instance.

Consider following cases depending on where your working machine locates.
 
## Direct Connect Considerations
If your working machine (machine for running CloudTik commands to manage clusters running on Cloud)
can directly connect cloud instances without proxy, it will be simple.

First, you don't need any proxy settings in cluster configuration file.
Make sure the 'ssh_proxy_command' in auth section is commented out. 
```
auth:
    ssh_user: ubuntu
    # Set proxy if you are in corporation network. For example,
    # ssh_proxy_command: "ncat --proxy-type socks5 --proxy your_proxy_host:your_proxy_port %h %p"
```

For cloud firewall settings for workspace configuration file, you usually have two options.

You can allow any IP addresses by allowing 0.0.0.0/0 to connect to cloud instances if your security requirements
allow it.

```
# Cloud-provider specific configuration.
provider:
    # Use allowed_ssh_sources to allow SSH access from your working machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```

You can also set the 'allowed_ssh_sources' under provider section with your public IP of your working machine.

```
# Cloud-provider specific configuration.
provider:
    # Use allowed_ssh_sources to allow SSH access from your working machine
    allowed_ssh_sources:
      - x.x.x.x/32
```

Replace 'x.x.x.x' with your working machine public ip.


## Behind Proxy Considerations
If your working machine is behind a proxy such as in the corporate network,
you need to configure the proxy and optionally the firewall settings.

For proxy, you need to set 'ssh_proxy_command' in cluster configuration file.
For example,
```
auth:
    ssh_user: ubuntu
    # Set proxy if you are in corporation network. For example,
    ssh_proxy_command: "ncat --proxy-type socks5 --proxy your_proxy_host:your_proxy_port %h %p"
```
Replace your_proxy_host and your_proxy_port with your own values.

For cloud firewall settings for workspace configuration file, you usually have two options.

You can allow any IP addresses by allowing 0.0.0.0/0 to connect to cloud instances if your security requirements
allow it.

```
# Cloud-provider specific configuration.
provider:
    # Use allowed_ssh_sources to allow SSH access from your working machine
    allowed_ssh_sources:
      - 0.0.0.0/0
```

If you want to restrict the IP addresses allowed to connect the cloud instances, 
you need set the proxy public IP address in the 'allowed_ssh_sources' under provider section.

```
# Cloud-provider specific configuration.
provider:
    # Use allowed_ssh_sources to allow SSH access from your working machine
    allowed_ssh_sources:
      - x.x.x.x/32
      - x.x.x.y/32
```

Replace 'x.x.x.x' with your proxy public IP address. If you have multiple proxies, you can list all of them.
