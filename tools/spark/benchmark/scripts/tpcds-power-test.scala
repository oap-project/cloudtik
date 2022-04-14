val conf = spark.sparkContext.getConf

// data scale in GB
val scaleFactor = conf.getInt("spark.driver.scaleFactor", 1)
// support parquet or orc
val format = conf.get("spark.driver.format", "parquet")
// create partitioned table
val partitionTables = conf.getBoolean("spark.driver.partitioned", true)
// how many times to run the whole set of queries.
val iterations = conf.getInt("spark.driver.iterations", 1)
// support s3a://s3_bucket, gs://gs_bucket
// wasbs://container@storage_account.blob.core.windows.net
// abfs://container@storage_account.dfs.core.windows.net
val fsdir = conf.get("spark.driver.fsdir", "")

val query_filter = Seq()        // Seq() == all queries
// val query_filter = Seq("q1-v2.4", "q2-v2.4") // run subset of queries
val randomizeQueries = false    // run queries in a random order. Recommended for parallel runs.

if (fsdir == "") {
    println("File system dir must be specified with --conf spark.driver.fsdir")
    sys.exit(0)
}

// detailed results will be written as JSON to this location.
var resultLocation = s"${fsdir}/shared/data/results/tpcds_${format}/${scaleFactor}/"
val data_path = s"${fsdir}/shared/data/tpcds/tpcds_${format}/${scaleFactor}"
var databaseName = s"tpcds_${format}_scale_${scaleFactor}_db"
val use_arrow = false            // when you want to use gazella_plugin to run TPC-DS, you need to set it true.

if (use_arrow){
    resultLocation = s"${fsdir}/shared/data/results/tpcds_arrow/${scaleFactor}/"
    databaseName = s"tpcds_arrow_scale_${scaleFactor}_db"
    val tables = Seq("call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer", "customer_address", "customer_demographics", "date_dim", "household_demographics", "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store", "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns", "web_sales", "web_site")
    sql(s"DROP database $databaseName CASCADE")
    if (spark.catalog.databaseExists(s"$databaseName")) {
        println(s"$databaseName has exists!")
    }else{
        spark.sql(s"create database if not exists $databaseName").show
        spark.sql(s"use $databaseName").show
        for (table <- tables) {
            if (spark.catalog.tableExists(s"$table")){
                println(s"$table has exists!")
            }else{
                spark.catalog.createTable(s"$table", s"$data_path/$table", "arrow")
            }
        }
        if (partitionTables) {
            for (table <- tables) {
                try{
                    spark.sql(s"ALTER TABLE $table RECOVER PARTITIONS").show
                }catch{
                        case e: Exception => println(e)
                }
            }
        }
    }
} else {
    // Check whether the database is created, we create external tables if not
    if (spark.catalog.databaseExists(s"$databaseName")) {
        println(s"Using existing $databaseName")
    } else {
        import com.databricks.spark.sql.perf.tpcds.TPCDSTables

        println(s"$databaseName doesn't exist. Creating...")
        val tables = new TPCDSTables(spark.sqlContext, "", s"${scaleFactor}", false)
        tables.createExternalTables(data_path, format, database_name, overwrite = true, discoverPartitions = partitionTables)
    }
}

val timeout = 60 // timeout in hours

// COMMAND ----------

// Spark configuration
spark.conf.set("spark.sql.broadcastTimeout", "10000") // good idea for Q14, Q88.

// ... + any other configuration tuning

// COMMAND ----------
sql(s"use $databaseName")
import com.databricks.spark.sql.perf.tpcds.TPCDS
val tpcds = new TPCDS (sqlContext = spark.sqlContext)
def queries = {
  val filtered_queries = query_filter match {
    case Seq() => tpcds.tpcds2_4Queries
    case _ => tpcds.tpcds2_4Queries.filter(q => query_filter.contains(q.name))
  }
  if (randomizeQueries) scala.util.Random.shuffle(filtered_queries) else filtered_queries
}
val experiment = tpcds.runExperiment(
  queries,
  iterations = iterations,
  resultLocation = resultLocation,
  tags = Map("runtype" -> "benchmark", "database" -> databaseName, "scale_factor" -> s"${scaleFactor}"))

println(experiment.toString)
experiment.waitForFinish(timeout*60*60)
sys.exit(0)
