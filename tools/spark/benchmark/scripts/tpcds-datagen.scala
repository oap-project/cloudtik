val conf = spark.sparkContext.getConf

// data scale in GB
val scale = conf.getInt("spark.driver.scale", 1)
// support parquet or orc
val format = conf.get("spark.driver.format", "parquet")
// create partitioned table
val partitionTables = conf.getBoolean("spark.driver.partitioned", true)
// support s3a://s3_bucket, gs://gs_bucket
// wasbs://container@storage_account.blob.core.windows.net
// abfs://container@storage_account.dfs.core.windows.net
val fsdir = conf.get("spark.driver.fsdir", "")
val useDoubleForDecimal = false   // use double format instead of decimal format

if (fsdir == "") {
    println("File system dir must be specified with --conf spark.driver.fsdir")
    sys.exit(0)
}

val tools_path = "/home/cloudtik/runtime/benchmark-tools/tpcds-kit/tools"
val data_path = s"${fsdir}/shared/data/tpcds/tpcds_${format}/${scale}"
val database_name = s"tpcds_${format}_scale_${scale}_db"
val codec = "snappy"
val clusterByPartitionColumns = partitionTables

val p = scale / 2048.0
val catalog_returns_p = (263 * p + 1).toInt
val catalog_sales_p = (2285 * p * 0.5 * 0.5 + 1).toInt
val store_returns_p = (429 * p + 1).toInt
val store_sales_p = (3164 * p * 0.5 * 0.5 + 1).toInt
val web_returns_p = (198 * p + 1).toInt
val web_sales_p = (1207 * p * 0.5 * 0.5 + 1).toInt

import com.databricks.spark.sql.perf.tpcds.TPCDSTables

val sc = spark.sqlContext
sc.setConf(s"spark.sql.$format.compression.codec", codec)

val tables = new TPCDSTables(spark.sqlContext, tools_path, scale, useDoubleForDecimal)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "call_center", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "catalog_page", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "customer", 6)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "customer_address", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "customer_demographics", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "date_dim", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "household_demographics", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "income_band", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "inventory", 6)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "item", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "promotion", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "reason", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "ship_mode", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "store", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "time_dim", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "warehouse", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "web_page", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "web_site", 1)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "catalog_sales", catalog_sales_p)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "catalog_returns", catalog_returns_p)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "store_sales", store_sales_p)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "store_returns", store_returns_p)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "web_sales", web_sales_p)
tables.genData(data_path, format, true, partitionTables, clusterByPartitionColumns, false, "web_returns", web_returns_p)
tables.createExternalTables(data_path, format, database_name, overwrite = true, discoverPartitions = partitionTables)

sys.exit(0)
