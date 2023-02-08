# Mount Cloud Storage

CloudTik provides the ability to easily and automatically mount cloud storage on your clusters. 
With this feature, you can access your existing cloud storage data through the Linux file system.
The supported cloud storage fuse include:
- [S3FS](https://github.com/s3fs-fuse/s3fs-fuse)
- [Blobfuse](https://github.com/Azure/azure-storage-fuse)
- [GCSUSE](https://github.com/GoogleCloudPlatform/gcsfuse)
- [HDFSFUSE](https://github.com/apache/hadoop/blob/trunk/hadoop-hdfs-project/hadoop-hdfs-native-client/src/main/native/fuse-dfs/doc/README)

## Notes
- The default mount directory is `/cloudtik/fs`.
- If you enable hdfs runtime, cloudtik will mount hdfs instead of other cloud storages.
- Due to the [issue](https://github.com/s3fs-fuse/s3fs-fuse/issues/1778) of S3FS, if you use kubernetes in AWS ECS, cloudtik can't automatically mount S3.
    We recommend you use hdfs fuse. Or you need to specify the **s3.bucket**, **s3.access.key.id** and **s3.secret.access.key** explicitly.



