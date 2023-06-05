# Configuring Cloud Database

- [Managed Cloud Database](#managed-cloud-database)
- [AWS RDS](#aws-rds)
- [Azure SQL](#azure-sql)
- [GCP Cloud SQL](#gcp-cloud-sql)


## Managed Cloud Database
To make database storage management and configuration simple for user, CloudTik offers two things:

- CloudTik can create a managed cloud database for you
if you set managed_cloud_database option when creating workspace.
- CloudTik can configure runtimes to use the cloud database
if you set use_managed_cloud_database option when starting cluster.

### Creating managed cloud database
When you create workspace, you can specify in workspace configuration
whether you need to create managed cloud database for the workspace.
Currently, we support only AWS, Azure and GCP managed cloud database.

Set managed_cloud_database to True in provider section of workspace config
for creating managed cloud database.

For example, in workspace configuration,

```buildoutcfg
# Cloud-provider specific configuration.
provider:
    # Set managed_cloud_database to True for creating managed cloud database 
    managed_cloud_database: True
```

### Using managed cloud database
If the managed cloud database is created for the workspace,
you can specify in cluster configuration
whether you want to use managed cloud database for runtimes
when you are creating a cluster.

For example, in cluster configuration,

```buildoutcfg
# Cloud-provider specific configuration.
provider:
    # Set use_managed_cloud_database to True for using managed cloud database 
    use_managed_cloud_database: True
```

## AWS RDS
If you accepted the default options for AWS RDS,
the only thing you need to configure is managed_cloud_database flag
in workspace configuration and use_managed_cloud_database in cluster configuration.

You only need to check this section if you want to customize.

### AWS RDS creation options
We have provided the proper default options when creating AWS RDS.
If you want to change,
you can specify the following options in the workspace configuration.

```buildoutcfg

provider:
    type: aws
    region: us-west-2
    database:
        # Database creating options
        aws.database:
            instance_type: AWS RDS DBInstanceClass. default: db.t3.xlarge
            storage_type: AWS RDS StorageType. default: gp2
            storage_size: AWS RDS AllocatedStorage size in GB. default: 50
            username: AWS RDS MasterUserPassword. default: cloudtik
            password: AWS RDS MasterUserPassword
```

### Configuring to use AWS RDS
We will automatically configure the right parameters for cluster
if you use the default options.
If you changed for example, username or password options above, 
You may need to update the cluster configurations to use the proper options.
You may also use the options here to configure to use other cloud database
you created by your own.

```
# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    database:
        # Database connecting options
        aws.database:
            server_address: AWS RDS server address. default: the managed cloud database server.
            port: AWS RDS server port. default: 3306
            username: AWS RDS MasterUserPassword. default: cloudtik
            password: AWS RDS MasterUserPassword

```

## Azure SQL
If you accepted the default options for Azure SQL,
the only thing you need to configure is managed_cloud_database flag
in workspace configuration and use_managed_cloud_database in cluster configuration.

You only need to check this section if you want to customize.

### Azure SQL creation options
We have provided the proper default options when creating Azure SQL.
If you want to change,
you can specify the following options in the workspace configuration.

```buildoutcfg

provider:
    type: azure
    location: westus
    subscription_id: your_subscription_id
    database:
        # Database creating options
        azure.database:
            instance_type: Azure Database InstanceSku. default: Standard_D4ds_v4
            storage_size: Azure Database storage size in GB. default: 50
            username: Azure Database administrator login name. default: cloudtik
            password: Azure Database administrator password
```

### Configuring to use Azure SQL
We will automatically configure the right parameters for cluster
if you use the default options.
If you changed for example, username or password options above, 
You may need to update the cluster configurations to use the proper options.
You may also use the options here to configure to use other cloud database
you created by your own.

```
# Cloud-provider specific configuration.
provider:
    type: azure
    location: westus
    subscription_id: your_subscription_id
    database:
        # Database connecting options
        azure.database:
            server_address: Azure Database server address. default: the managed cloud database server.
            port: Azure Database server port. default: 3306
            username: Azure Database administrator login name. default: cloudtik
            password: Azure Database administrator login password
```

## GCP Cloud SQL
If you accepted the default options for GCP Cloud SQL,
the only thing you need to configure is managed_cloud_database flag
in workspace configuration and use_managed_cloud_database in cluster configuration.

You only need to check this section if you want to customize.

### GCP Cloud SQL creation options
We have provided the proper default options when creating GCP Cloud SQL.
If you want to change,
you can specify the following options in the workspace configuration.

```buildoutcfg

provider:
    type: gcp
    region: us-central1
    availability_zone: us-central1-a
    project_id: your_project_id
    database:
        # Database creating options
        azure.database:
            instance_type: GCP Cloud SQL machine type. default: db-custom-4-15360
            storage_type: GCP Cloud SQL storage type. default: PD_SSD
            storage_size: GCP Cloud SQL storage size in GB. default: 50
            password: GCP Cloud SQL root password
```

### Configuring to use GCP Cloud SQL
We will automatically configure the right parameters for cluster
if you use the default options.
If you changed for example, username or password options above, 
You may need to update the cluster configurations to use the proper options.
You may also use the options here to configure to use other cloud database
you created by your own.

```
# Cloud-provider specific configuration.
provider:
    type: gcp
    region: us-central1
    availability_zone: us-central1-a
    project_id: your_project_id
    database:
        # Database connecting options
        azure.database:
            server_address: GCP Cloud SQL server address. default: the managed cloud database server.
            port: GCP Cloud SQL server port. default: 3306
            username: GCP Cloud SQL user name. default: root
            password: GCP Cloud SQL password
```
