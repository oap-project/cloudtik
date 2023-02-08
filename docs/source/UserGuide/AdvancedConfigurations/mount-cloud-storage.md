# Mount Cloud Storage
CloudTik Spark runtime support to mount cloud storage to local path through Fuse.
This is automatically done by CloudTik on your clusters without any configurations
as long as the Spark runtime is configured and is able to access the cloud storage.
With the local mount of cloud storage, you can access your existing cloud storage data through the Linux file system
which makes your existing code easier to be ported to CloudTik runtime environment.

The path to which CloudTik will mount the cloud storage is `/cloudtik/fs`.

If you enable hdfs runtime, cloudtik will mount hdfs instead of other cloud storages.

## Limitations or Known Issues
CloudTik uses the fuse implementations for the corresponding cloud storage
to do the work.
- [s3fs-fuse](https://github.com/s3fs-fuse/s3fs-fuse)
- [azure-storage-fuse](https://github.com/Azure/azure-storage-fuse)
- [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse)
- [fuse-dfs](https://github.com/apache/hadoop/blob/trunk/hadoop-hdfs-project/hadoop-hdfs-native-client/src/main/native/fuse-dfs/doc/README)

These libraries may have limitations for specific authentication scenarios.

### s3fs-fuse Limitations
Due to the [issue](https://github.com/s3fs-fuse/s3fs-fuse/issues/1778) of s3fs-fuse,
if you use kubernetes in AWS ECS, cloudtik can't automatically mount S3.
You need to specify the **s3.access.key.id** and **s3.secret.access.key** explicitly
in configuration file explicitly for this case to work.

### azure-storage-fuse Limitations
Azure storage fuse cannot work with Azure AKS workload identity ([issue](https://github.com/Azure/azure-storage-fuse/issues/1049)).
And thus cloudtik can't automatically mount when workload identity for accessing cloud storage is used.
You can specify the credential information through azure_cloud_storage configurations
in configuration file explicitly for this case to work.
